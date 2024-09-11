import torch
import torch.nn as nn
import numpy as np
import copy
import wandb

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from lora_and_ia3 import Lora_Wrapper, IA3_Wrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler


def train(
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        train_loader: DataLoader, 
        validation_loaders: dict,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR, 
        progress_bar: tqdm,
        device: str = "cuda",
        eval_steps: int = 2000,
        run: wandb = None,
        early_stopping_patience: int = 3,
        epochs: int = 10
    ):
    """Training function for the model
    
    Args:
        model (nn.Module): The model
        optimizer (torch.optim.Optimizer): The optimizer
        train_loader (DataLoader): The training dataloader
        validation_loaders (dict): The validation dataloaders
        lr_scheduler (torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler
        progress_bar (tqdm): The progress bar
        device (str, optional): The device to use. Defaults to "cuda".
        eval_steps (int, optional): The number of steps to evaluate the model. Defaults to 2000.
        run (wandb, optional): The wandb run. Defaults to None.
        early_stopping_patience (int, optional): The patience for early stopping. Defaults to 3.
        epochs (int, optional): The number of epochs. Defaults to 10.
    
    Returns:
        nn.Module: The best model
    """

    step = 0
    best_quy_Latn_loss = float("inf")
    best_model = copy.deepcopy(model)
    global_running_train_loss = []

    model.to(device)
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Enables autocasting for the forward pass (model + loss)
            with torch.autocast(device_type="cuda"):
                outputs = model(**batch)
                loss = outputs.loss
                global_running_train_loss.append(loss.cpu().item())
                wandb.log({"train_loss": loss.cpu().item()})

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            step += 1

            # Doing some cleanup
            del loss
            del batch
            del outputs
            torch.cuda.empty_cache()
            
            # Based on the eval_steps, we will evaluate the model
            if step % eval_steps == 0:
                print("Evaluating model...")
                train_loss = np.mean(global_running_train_loss)
                global_running_train_loss = []
                model.eval()

                validation_losses = {}
                for key in validation_loaders.keys():
                    validation_losses[key] = validate(model, validation_loaders[key], language=key)   
                    run.log({f"val_loss_{key}": validation_losses[key]})
                
                print(f"Epoch {epoch + 1}, step {step}: Train loss: {train_loss:.3f}, Val loss: {validation_losses}")

                if (validation_losses["quy_Latn"] < best_quy_Latn_loss):
                    best_quy_Latn_loss = validation_losses["quy_Latn"]
                    best_model = copy.deepcopy(model)
                    patience = 0
                else:
                    patience += 1

                if patience > early_stopping_patience:
                    return best_model

    return best_model

def validate(model: nn.Module, dataloader: DataLoader, device:str = "cuda", language: str = "quy_Latn"):
    """Validation function for the model
    
    Args:
        model (nn.Module): The model
        dataloader (DataLoader): The dataloader
        device (str, optional): The device to use. Defaults to "cuda".
        language (str, optional): The language to validate on. Defaults to "quy_Latn".
    
    Returns:
        float: The validation loss
    """
    model.eval()
    running_loss = []

    print(f"Validating on language: {language}")
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.autocast(device_type="cuda"):
                outputs = model(**batch)
                running_loss.append(outputs.loss.cpu().item())

            del batch
            del outputs
            torch.cuda.empty_cache()

    mean_loss = np.mean(running_loss)
    print(f"Validation loss: {mean_loss}")
    return mean_loss

def training_loop(
    train_set,
    tokenizer: AutoTokenizer,
    validation_sets, 
    model_name: str,
    epochs: int, 
    train_batch_size: int,
    validation_batch_size: int,
    learning_rate: float = 5e-5, 
    early_stopping_patience: int = 3,
    method: str = "full",
    run: wandb = None,
    device: str = "cuda",
    save_path: str = "model.pt"
):
    """Training loop for fine-tuning a model

    Args:
        train_set (Dataset): The training dataset
        tokenizer (AutoTokenizer): The tokenizer
        validation_sets (dict): A dictionary of validation datasets
        model_name (str): The model name
        epochs (int): The number of epochs
        train_batch_size (int): The batch size for training
        validation_batch_size (int): The batch size for validation
        learning_rate (float, optional): The learning rate. Defaults to 5e-5.
        early_stopping_patience (int, optional): The patience for early stopping. Defaults to 3.
        method (str, optional): The fine-tuning method. Defaults to "full".
        run (wandb, optional): The wandb run. Defaults to None.
        device (str, optional): The device to use. Defaults to "cuda".
        save_path (str, optional): The path to save the model. Defaults to "model.pt".
    
    Returns:
        None
    """
    train_loader = DataLoader(
        train_set, 
        shuffle=True, 
        batch_size=train_batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    validation_loaders = {}
    for key in validation_sets.keys():
        validation_loaders[key] = DataLoader(
            validation_sets[key],
            batch_size=validation_batch_size,
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply fine-tuning method
    if method == "full":
        print("Full fine tuning...")
        apply_full_fine_tune(model)
    elif method == "bitfit":
        print("BitFiting...")
        apply_bitfit(model)
    elif method == "lora":
        print("LoRa-ing")
        rank = 16
        print("rank: ", rank)
        Lora = Lora_Wrapper(rank=rank)
        model = Lora.get_model()
    elif method == "ia3":
        print("IA3-ing")
        ia3 = IA3_Wrapper()
        model = ia3.get_model()
    else:
        print("Unknown method")
        exit(1)
    
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps), desc="Training")

    best_model = train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loaders=validation_loaders,
        lr_scheduler=lr_scheduler,
        progress_bar=progress_bar,
        device=device,
        run=run,
        early_stopping_patience=early_stopping_patience,
        epochs=epochs
    )

    print("Training finished. Saving model...")
    torch.save(best_model, save_path)

def collate_fn(batch, tokenizer):
    """Collate function to convert a batch of samples into a batch of padded tokenized sequences

    Args:
        batch (list): a list of samples
        tokenizer (Tokenizer): the tokenizer

    Returns:
        dict: a dictionary of tokenized sequences
    """
    # For Spanish-to-quechua dataset
    if "es" in batch[0] and "qu" in batch[0]:
        return tokenizer.tokenize([sample["qu"] for sample in batch])

    # For Flores dataset
    if "sentence" in batch[0]:
        return tokenizer.tokenize([sample["sentence"] for sample in batch])

    # For NLLB dataset
    if "translation" in batch[0]:
        return tokenizer.tokenize([sample["translation"]["quy_Latn"] for sample in batch])

    print("Unsupported dataset!")
    exit(1)

def apply_full_fine_tune(model):
    """Applies full fine-tuning to the given model

    Args:
        model (AutoModelForCausalLM): The model to apply full fine-tuning to
    
    Returns:
        None
    """
    for _, param in model.lm_head.named_parameters():
        param.requires_grad = True

def apply_bitfit(model, keep=["bias"]):
    """Applies BitFit to the given model

    Args:
        model (AutoModelForCausalLM): The model to apply BitFit to
        keep (list, optional): A list of parameter names to keep. Defaults to ["bias"].

    Returns:
        None
    """
    # Make sure to not freeze the weights of the last layer
    for name, param in model.named_parameters():
        param.requires_grad = any(en in name for en in keep)

    # Make sure to include attiontion head to be fine-tuned
    for _, param in model.lm_head.named_parameters():
        param.requires_grad = True