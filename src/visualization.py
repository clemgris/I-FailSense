import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def visualization_report(y_true, y_pred, model_name, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    # Compute confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["fail", "success"],
        zero_division=0,
        digits=4,
    )

    # Save results to .txt
    txt_path = os.path.join(output_dir, f"{model_name}_results.txt")
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"[✓] Metrics saved to {txt_path}")

    # Save confusion matrix image
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["fail", "success"],
        yticklabels=["fail", "success"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    img_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(img_path)
    plt.close()

    print(f"[✓] Confusion matrix image saved to {img_path}")
