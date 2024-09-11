def is_in_range(text, lower=35, upper=434):
    """Checks if the length of the given text is in the given range

    Args:
        text (str): The text to check
        lower (int, optional): The lower bound of the range. Defaults to 35.
        upper (int, optional): The upper bound of the range. Defaults to 434.
    
    Returns:
        bool: True if the length of the text is in the given range, False otherwise
    """
    return lower <= len(text) <= upper

def preprocess(dataset, path):
    """Preprocesses the given dataset

    Args:
        dataset (Dataset): The dataset to preprocess

    Returns:
        Dataset: The preprocessed dataset
    """
    # Make sure that the sentence length is in specific range
    if path == "hackathon-pln-es/spanish-to-quechua":
        return dataset.filter(
            lambda example: is_in_range(example["qu"])
        )
    return dataset.filter(
        lambda example: is_in_range(example["translation"]["quy_Latn"])
    )