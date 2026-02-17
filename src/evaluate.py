import argparse
import gc
import os

import torch
from inference import batch_inference
from load_dataset import augment_droid_dataset, load_data
from model import FailSense
from visualization import visualization_report

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


def evaluate_model_on_dataset(
    model,
    dataset_name,
    vlm_model_id,
    checkpoint_name,
    batch_size=8,
    result_folder=None,
):
    model_name = vlm_model_id.split("/")[-1]
    style = "video" if "Video" in model_name else "image"
    pov = 1 if "1p" in model_name else 2

    # Load dataset
    test_sample = load_data(
        dataset_name=dataset_name, split="test", num_entry="full", style=style, pov=pov
    )

    if dataset_name == "droid":
        # Create SM failures for droid
        test_sample = augment_droid_dataset(test_sample)

    # Inference
    predictions, labels = batch_inference(model, test_sample, batch_size=batch_size)

    # Normalize predictions and labels
    labels = [1 if label in (1, "success", "1") else 0 for label in labels]
    predictions = [1 if pred in (1, "success", "1") else 0 for pred in predictions]

    # Generate report
    report_name = f"{dataset_name}-{model_name}-{checkpoint_name}"
    visualization_report(
        labels,
        predictions,
        model_name=report_name,
        output_dir=f"./results/{dataset_name}/{model_name}"
        if result_folder is None
        else result_folder,
    )


def main(args):
    batch_size = 4
    fs_dir = args.fs_id

    for checkpoint_name in os.listdir(fs_dir):
        if not checkpoint_name.endswith(".pt"):
            pass  # do nothing
        else:
            checkpoint_path = os.path.join(fs_dir, checkpoint_name)

            # Load model and checkpoint
            model = FailSense(args.vlm_model_id, device=device)
            model.load_classifier(checkpoint_path)
            model.eval()

            evaluate_model_on_dataset(
                model,
                args.dataset_name,
                args.vlm_model_id,
                checkpoint_name,
                batch_size=batch_size,
            )

            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate FailSense model on datasets")
    parser.add_argument(
        "--vlm_model_id",
        type=str,
        default=None,
        choices=[
            "ACIDE/FailSense-Calvin-1p-3b",
            "ACIDE/FailSense-Calvin-2p-3b",
        ],
        help="Base model ID to evaluate",
    )

    parser.add_argument(
        "--fs_id",
        type=str,
        default=None,
        choices=[
            "FS/FailSense-Calvin-1p-3b",
            "FS/FailSense-Calvin-2p-3b",
            "FS/FailSense-DROID-1p-3b",
            "FS/FailSense-DROID-2p-3b",
        ],
        help="FS blocks ID to evaluate",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        choices=[
            "droid",
            "calvin",
            "aha",
        ],
        help="Dataset name to evaluate on",
    )

    parser.add_argument(
        "--result_folder",
        type=str,
        default=None,
        help="Custom folder to save results (overrides default structure)",
    )

    args = parser.parse_args()

    main(args)
