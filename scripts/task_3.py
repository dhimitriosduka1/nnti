import os
import sys
import yaml
import time
import torch
import wandb


from model import training_loop
from tokenizer import Tokenizer
from preprocess import preprocess
from datasets import load_dataset

if __name__ == "__main__":

    # Check if the config file is provided
    if len(sys.argv) < 2:
        print("USAGE: python task_3.py <CONFIG_FILE>")
        exit(1)

    # Load the configuration file
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)
        METHOD = config["method"]
        WANDB_PROJECT = config["wandb_project"]
        PER_DEVICE_TRAIN_BATCH_SIZE = config["per_device_train_batch_size"]
        PER_DEVICE_EVAL_BATCH_SIZE = config["per_device_eval_batch_size"]
        EPOCHS = config["epochs"]

        TRAIN_DATASET_CONFIG = config["dataset"]["train"]
        VALIDATION_DATASET_CONFIG = config["dataset"]["validation"] if "validation" in config["dataset"] else None

        TOKENIZER_CONFIG = config["tokenizer"]
        LEARNING_RATE = config["learning_rate"]

    run = wandb.init(
        # Set the project where this run will be logged
        project=WANDB_PROJECT,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "method": METHOD
        },
    )
    

    #  Get the path for saving the model
    dir_path = str(os.path.dirname(os.path.realpath(__file__)))
    dir_path = dir_path.replace("/scripts", "")

    # Set the model name
    MODEL_NAME = "facebook/xglm-564M"

    # Define the number of samples in training dataset
    NUM_TRAIN_SAMPLES = 80000

    # Languages
    LANGUAGES = [
        "eng_Latn",
        "spa_Latn",
        "ita_Latn",
        "deu_Latn",
        "arb_Arab",
        "tel_Telu",
        "tam_Taml",
        "quy_Latn"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")


    # Create a tokenizer instance for the given model name with the given configuration
    tokenizer = Tokenizer(MODEL_NAME)

    # Load the dataset
    train_dataset = load_dataset(
        TRAIN_DATASET_CONFIG["path"],
        TRAIN_DATASET_CONFIG["name"],
        trust_remote_code=True
    )

    # If a split is provided, use the split
    if "split" in TRAIN_DATASET_CONFIG:
        train_dataset = train_dataset[TRAIN_DATASET_CONFIG["split"]]

    print(f"FEATURRE COLUMNS OF TRAINING DATASET:\n {train_dataset.info.features}\n")
    print(f"LENGTH OF TRAINING DATASET BEFORE PREPROCESSING: {len(train_dataset)}\n")

     # Preprocess the dataset
    train_dataset = preprocess(train_dataset, TRAIN_DATASET_CONFIG["path"])
    print(f"LENGTH OF TRAINING DATASET AFTER PREPROCESSING: {len(train_dataset)}\n")

    # Reduce the number of samples in the training dataset to NUM_TRAIN_SAMPLES
    train_dataset = train_dataset.shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
    print(f"LENGTH OF TRAINING DATASET AFTER SAMPLING: {len(train_dataset)}\n")

    # Load the validation datasets
    eval_datasets = {}
    for lang in LANGUAGES:
        eval_datasets[lang] = load_dataset(
            "facebook/flores",
            lang,
            trust_remote_code=True
        )["dev"]

        # If another validation datset is provided, use it
    if VALIDATION_DATASET_CONFIG is not None:
        eval_datasets["other"] = load_dataset(
            VALIDATION_DATASET_CONFIG["path"],
            VALIDATION_DATASET_CONFIG["name"],
            trust_remote_code=True
        )[VALIDATION_DATASET_CONFIG["split"]]

    dataset_name = TRAIN_DATASET_CONFIG["path"]
    now = str(time.time() * 1000).split(".")[0]

    model_save_path = f"{dir_path}/models/{METHOD}/{dataset_name}/{now}/"

    # Check if the directory is exists
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    training_loop(
        train_set=train_dataset,
        tokenizer=tokenizer,
        validation_sets=eval_datasets,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        validation_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        method=METHOD,
        run=run,
        device=device,
        save_path=f"{model_save_path}/model.pt"
    )