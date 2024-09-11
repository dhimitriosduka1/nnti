import torch

from transformers import AutoModelForCausalLM
from torch.nn import functional as F
from typing import Optional, Tuple
from transformers import XGLMConfig
from transformers.models.xglm.modeling_xglm import XGLMAttention
from transformers.models.xglm.modeling_xglm import XGLMDecoderLayer

class Lora_Self_Attention(XGLMAttention):
    def __init__(self, r=8, embed_dim=0, num_heads=0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)    
        d = self.embed_dim                                         

        # Initialize attention matrices
        self.lora_query_matrix_A = torch.nn.Parameter(torch.randn(r, d))
        self.lora_value_matrix_A = torch.nn.Parameter(torch.rand(r,d))

        # Initialize B matrices to zero, so that the fine-tuning begins with the original XGLM-564M
        self.lora_query_matrix_B = torch.nn.Parameter(torch.zeros(d, r))
        self.lora_value_matrix_B = torch.nn.Parameter(torch.zeros(d,r))

    def lora_q_proj(self,x):
        lora_query_matrix = torch.matmul(self.lora_query_matrix_B, self.lora_query_matrix_A)
        return self.q_proj(x)+  F.linear(x, lora_query_matrix)
    
    def lora_v_proj(self,x):
        lora_value_matrix = torch.matmul(self.lora_value_matrix_B, self.lora_value_matrix_A)
        return self.v_proj(x)+  F.linear(x, lora_value_matrix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.lora_q_proj(hidden_states) * self.scaling                       # replaced self.q_proj 
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.lora_v_proj(key_value_states), -1, bsz)          # replaced self.v_proj
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.lora_v_proj(hidden_states), -1, bsz)             # replaced self.v_proj
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.lora_v_proj(hidden_states), -1, bsz)             # replaced self.v_proj

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class Lora_Wrapper():
    def __init__(self, model_name= "facebook/xglm-564M", rank= 8):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.rank = rank
        self.replace_Attention(self.model)
        self.freeze_weights()
        self.unfreeze_lm_head()

    def get_model(self):
        return self.model
        

    def replace_Attention(self, model):
        # Iterate and only replace XGLMAttention
        for name, module in model.named_children():
            if isinstance(module, XGLMAttention):
                attention_replacement = Lora_Self_Attention(r= self.rank, embed_dim= self.model.config.d_model ,num_heads= self.model.config.attention_heads,)#, config=self.model.config)
                attention_replacement.load_state_dict(module.state_dict(), strict=False)
                setattr(model, name, attention_replacement)
            else:
                self.replace_Attention(module)

    def freeze_weights(self, enable = ['lora_query_matrix', 'lora_value_matrix']): 
        for name, param in self.model.named_parameters():
            param.requires_grad = any(en in name for en in enable)
            
    def unfreeze_lm_head(self):
        for parameter in self.model.lm_head.parameters():        
            parameter.requires_grad = True
            
class IA3_Self_Attention(XGLMAttention):
    def __init__(self, embed_dim=0, num_heads=0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)    
        d = self.embed_dim                                           

        # Initialize attention matrices
        self.IA3_l_k= torch.nn.Parameter(torch.ones(d))    # key
        self.IA3_l_v= torch.nn.Parameter(torch.ones(d))    # value
    
    def IA3_k_proj(self,x):
        return torch.mul(self.IA3_l_k, self.k_proj(x))
    
    def IA3_v_proj(self,x):
        return torch.mul(self.IA3_l_v, self.v_proj(x))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling                
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.IA3_k_proj(key_value_states), -1, bsz)          # replaced self.k_proj
            value_states = self._shape(self.IA3_v_proj(key_value_states), -1, bsz)          # replaced self.v_proj
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.IA3_k_proj(hidden_states), -1, bsz)                 # replaced self.k_proj
            value_states = self._shape(self.IA3_v_proj(hidden_states), -1, bsz)            # replaced self.v_proj
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.IA3_k_proj(hidden_states), -1, bsz)                 # replaced self.k_proj
            value_states = self._shape(self.IA3_v_proj(hidden_states), -1, bsz)              # replaced  self.k_proj

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
    
class IA3_XGLMDecoderLayer(XGLMDecoderLayer):
    def __init__(self, config: XGLMConfig):
        super().__init__(config)
        d = config.ffn_dim
        self.IA3_l_ff = torch.nn.Parameter(torch.ones(d))   

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/xglm/modeling_xglm.py
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = torch.mul(self.IA3_l_ff, hidden_states) # included l_ff
        hidden_states = self.fc2(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class IA3_Wrapper():
    def __init__(self, model_name= "facebook/xglm-564M"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.replace_decoder(self.model)
        self.replace_Attention(self.model)
        # self.unfreeze_lm_head()
        self.freeze_weights()
    
    def get_model(self):
        return self.model

    def replace_decoder(self, model):
        for name, module in model.named_children():
            if (isinstance(module, XGLMDecoderLayer)):
                decoder_replacement = IA3_XGLMDecoderLayer(self.model.config)
                decoder_replacement.load_state_dict(module.state_dict(), strict=False)
                setattr(model, name, decoder_replacement)
            else:
                self.replace_decoder(module)

    def replace_Attention(self, model):
        # iterate and only replace XGLMAttention
        for name, module in model.named_children():
            if isinstance(module, XGLMAttention):
                attention_replacement = IA3_Self_Attention(embed_dim= self.model.config.d_model ,num_heads= self.model.config.attention_heads)
                attention_replacement.load_state_dict(module.state_dict(), strict=False)
                setattr(model, name, attention_replacement)
            else:
                self.replace_Attention(module)

    def freeze_weights(self, enable = ['IA3_l_k', 'IA3_l_v', 'IA3_l_ff']): 
        for name, param in self.model.named_parameters():
            param.requires_grad = any(en in name for en in enable)
      
    def unfreeze_lm_head(self):
        for parameter in self.model.lm_head.parameters():        
            parameter.requires_grad = True