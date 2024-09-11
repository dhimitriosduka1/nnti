import torch
from transformers import AutoTokenizer

class Tokenizer:
    """Tokenizer class to tokenize a given example for a given model
    """

    def __init__(self, model_name, padding="longest", truncation="longest_first", return_tensors="pt"):
        self.model_name = model_name
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text):
        """Tokenizes the given input text

        Args:
            example (str): The input text to be tokenized

        Returns:
            dict: A dictionary containing the tokenized input text, attention mask, and labels
        """
        tokenized = self.tokenizer(
            text,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=self.truncation
        )

        tokenized["labels"] = torch.where(
            tokenized["input_ids"] == self.tokenizer.pad_token_id,
            -100,
            tokenized["input_ids"]
        )
        return tokenized