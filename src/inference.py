import torch
import tqdm
from model import process_input


def batch_inference(model, dataset, batch_size):
    predictions = []
    labels = []

    total_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batches...")

    # Process dataset in batches
    for batch_idx, i in enumerate(
        tqdm.tqdm(range(0, len(dataset), batch_size), desc="Processing batches")
    ):
        batch_end = min(i + batch_size, len(dataset))
        entries = dataset[i:batch_end]
        batch_images = entries["images"]
        batch_texts = [
            process_input(entries["images"][z], entries["task"][z])
            for z in range(len(batch_images))
        ]
        batch_labels = [
            0 if entry in ("0", "fail", 0) else 1 for entry in entries["label"]
        ]

        batch_preds, _ = model.predict(batch_images, batch_texts, voting=True)

        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.view(-1).cpu().tolist()
        elif isinstance(batch_preds, int) or isinstance(batch_preds, float):
            batch_preds = [batch_preds]

        predictions.extend(batch_preds)

        labels.extend(batch_labels)

    print("-" * 50)
    print("Inference completed!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Total labels: {len(labels)}")

    return predictions, labels
