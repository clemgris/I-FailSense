import argparse
import os

import torch
from load_dataset import augment_droid_dataset, load_data
from model import FailSense, train_model


def main(args):
    # Parse model configuration from name
    model_name = args.vlm_model_id.split("/")[-1]
    parts = model_name.split("-")
    style = "video" if "Video" in parts else "image"
    pov = 1 if "1p" in parts else 2

    # Training configuration
    config = {
        "lr": 1e-4,
        "weight_decay": 0.1,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "validation_step": 100000,
        "save_path": f"./FS/{args.vlm_model_id}/{args.dataset_name}"
        if args.result_folder is None
        else args.result_folder,
        "dropout_rate": 0.5,
        "num_classifiers": 3,
        "vlm_model_id": args.vlm_model_id,
        "dataset_name": args.dataset_name,
    }

    # Save training config
    config_path = f"{config['save_path']}/config.txt"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Initialize model
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = FailSense(
        args.vlm_model_id,
        device=device,
        dropout_rate=config["dropout_rate"],
        num_classifiers=config["num_classifiers"],
    )

    ### Load datasets ###

    # Load sucesses
    train_dataset = load_data(
        dataset_name=args.dataset_name, style=style, split="train", pov=pov
    )
    # Create failures
    train_dataset = augment_droid_dataset(train_dataset)

    # Shuffle the dataset
    shuffled = train_dataset.shuffle(seed=42)
    train_dataset = shuffled
    # Select a small validation set from the shuffled dataset
    val_dataset = shuffled.select(range(1000, 1010))

    try:
        # Train model
        train_model(model, train_dataset, val_dataset, config)
    finally:
        # Cleanup
        model.cleanup()
        print("Model cleanup completed")

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate FailSense model on datasets")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="droid",
        choices=["droid", "calvin"],
        help="Dataset to train on",
    )

    parser.add_argument(
        "--result_folder",
        type=str,
        default=None,
        help="Custom folder to save FS checkpoints",
    )

    parser.add_argument(
        "--vlm_model_id",
        type=str,
        default=None,
        choices=[
            "ACIDE/FailSense-Calvin-1p-3b",
            "ACIDE/FailSense-Calvin-2p-3b",
        ],
        help="Base model ID",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    main(args)
