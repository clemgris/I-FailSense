import gc
import os

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from load_dataset import load_data
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

torch.manual_seed(42)


def process_input(images, text):
    num_images = len(images) if isinstance(images, list) else 1
    prompt = " ".join(["<image>"] * num_images) + " evaluate en " + text
    return prompt


def validate_model(model, val_dataset, batch_size):
    """Validation function"""
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i in range(0, len(val_dataset), batch_size):
            batch_end = min(i + batch_size, len(val_dataset))
            entries = val_dataset[i:batch_end]
            try:
                batch_images = entries["images"]
                batch_texts = [
                    process_input(entries["images"][z], entries["task"][z])
                    for z in range(len(batch_images))
                ]
                batch_labels = [
                    0 if entry in ("0", "fail", 0) else 1 for entry in entries["label"]
                ]
                predictions, avg_probs = model.predict(
                    batch_images, batch_texts, voting=True
                )
                batch_labels = torch.tensor(
                    batch_labels, dtype=torch.float32, device=model.device
                )
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)
            except Exception as e:
                print(f"[Warning] Skipping validation batch due to error: {e}")
                continue

    val_acc = val_correct / val_total if val_total > 0 else 0
    return val_acc


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, D]
        scores = self.attn(x)  # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # [B, T, 1]
        pooled = torch.sum(weights * x, dim=1)  # [B, D]
        return pooled


class HybridAttentionPooling(nn.Module):
    def __init__(
        self, input_dim, num_heads=4, hidden_dim=None, return_weights=False, dropout=0.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.return_weights = return_weights

        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable fusion weight
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # Boolean mask: True = pad
        attn_mask = ~mask.bool() if mask is not None else None
        mask_expanded = mask.unsqueeze(-1) if mask is not None else None

        # ====== 1) MLP scores ======
        mlp_scores = self.attn_mlp(x)  # [B, T, 1]

        # ====== 2) MHA scores ======
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        _, mha_weights = self.mha(query, x, x, key_padding_mask=attn_mask)
        mha_scores = mha_weights.transpose(1, 2)  # [B, T, 1]

        # ====== 3) Combine ======
        combined_scores = self.alpha * mlp_scores + (1 - self.alpha) * mha_scores
        combined_scores -= combined_scores.max(dim=1, keepdim=True).values

        if mask_expanded is not None:
            combined_scores = combined_scores.masked_fill(
                ~mask_expanded, torch.finfo(x.dtype).min
            )

        # ====== 4) Weights & pooling ======
        weights = F.softmax(combined_scores, dim=1)
        weights = self.attn_dropout(weights)
        pooled = torch.sum(weights * x, dim=1)

        return (pooled, weights.squeeze(-1)) if self.return_weights else pooled


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.4):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.act(x)
        x = self.fc(x)
        x = self.dropout(x)
        return residual + x


class MLP_BLOCK(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate=0.4):
        super().__init__()
        self.residual = ResidualBlock(dim_in, dropout_rate)
        self.norm = nn.BatchNorm1d(dim_in)
        self.act = nn.GELU()
        self.fc = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.residual(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x


class FailSense(nn.Module):
    def __init__(
        self,
        vlm_model_id: str,
        feature_dim: int = 2304,
        device: str = "cuda",
        dropout_rate: float = 0.3,
        num_classifiers: int = 3,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_classifiers = num_classifiers
        self.dropout_rate = dropout_rate

        # Load VLM components
        self.processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        # Load base model and PEFT adapter with memory optimization
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma2-3b-mix-224",
            device_map=self.device,
            torch_dtype=torch.float32,
        )

        print(f"Loading PEFT adapter from checkpoint {vlm_model_id}")

        self.vlm_model = PeftModel.from_pretrained(base_model, vlm_model_id)
        self.vlm_model.eval()  # Always keep VLM in eval mode

        for param in self.vlm_model.parameters():
            param.requires_grad = False

        self.att_poolings = nn.ModuleList(
            [
                HybridAttentionPooling(feature_dim).to(self.device)
                for _ in range(self.num_classifiers)
            ]
        )

        # Create classifiers with consistent dropout
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    MLP_BLOCK(feature_dim, 1024, dropout_rate),
                    MLP_BLOCK(1024, 256, dropout_rate),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, 1),
                ).to(self.device)
                for _ in range(self.num_classifiers)
            ]
        )

        self.layer_features = {}
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for intermediate layer feature extraction"""
        # Find the language model layers
        layers = self._find_model_layers()

        if layers is None:
            raise RuntimeError("Could not find model layers for hook registration")

        total_layers = len(layers)
        middle_idx = total_layers // 2

        # Select layers from middle to end with even spacing
        target_indices = []
        remaining_layers = total_layers - middle_idx
        step = max(1, remaining_layers // self.num_classifiers)

        for i in range(self.num_classifiers):
            if i == self.num_classifiers - 1:  # Last classifier uses the final layer
                target_indices.append(total_layers - 1)
            else:
                target_indices.append(min(middle_idx + i * step, total_layers - 1))

        # Register hooks for selected layers
        for i, layer_idx in enumerate(target_indices):
            target_layer = layers[layer_idx]

            def create_hook_fn(classifier_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        self.layer_features[classifier_idx] = output[
                            0
                        ].detach()  # Detach to prevent gradient flow
                    else:
                        self.layer_features[classifier_idx] = output.detach()

                return hook_fn

            hook_handle = target_layer.register_forward_hook(create_hook_fn(i))
            self.hook_handles.append(hook_handle)

    def _find_model_layers(self):
        """Find the transformer layers in the model architecture"""
        if hasattr(self.vlm_model.base_model.model.model, "language_model"):
            language_model = self.vlm_model.base_model.model.model.language_model
            if hasattr(language_model, "model") and hasattr(
                language_model.model, "layers"
            ):
                return language_model.model.layers
            elif hasattr(language_model, "layers"):
                return language_model.layers

        # Fallback approaches
        if hasattr(self.vlm_model.base_model.model.model, "layers"):
            return self.vlm_model.base_model.model.model.layers
        elif hasattr(self.vlm_model.base_model.model.model, "transformer"):
            if hasattr(self.vlm_model.base_model.model.model.transformer, "h"):
                return self.vlm_model.base_model.model.model.transformer.h

        return None

    def extract_features(self, images, text_prompts, voting=False):
        """Extract features from VLM model"""
        # Process inputs with proper error handling
        try:
            model_inputs = self.processor(
                text=text_prompts,
                images=images,
                return_tensors="pt",
                padding="longest",
                truncation=True,  # Add truncation
                max_length=512,  # Set reasonable max length
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error processing inputs: {e}")

        # Clear previous features - CRITICAL for memory management
        self.layer_features.clear()

        with torch.no_grad():
            vlm_output = self.vlm_model(**model_inputs)

        decoded = None
        if voting:
            logits = vlm_output.logits
            # Handle potential shape issues
            if len(logits.shape) >= 2:
                last_token_logits = logits[:, -1, :]
                predicted_token_id = torch.argmax(last_token_logits, dim=-1)
                decoded = [
                    self.processor.decode([token_id.item()], skip_special_tokens=True)
                    for token_id in predicted_token_id
                ]
            else:
                decoded = [""] * logits.shape[0]

        # Validate we got all expected features
        if len(self.layer_features) != self.num_classifiers:
            raise RuntimeError(
                f"Expected {self.num_classifiers} features, got {len(self.layer_features)}. "
                f"Check hook registration."
            )

        # Return features ensuring they're on the correct device
        features = [
            self.layer_features[i].to(self.device) for i in range(self.num_classifiers)
        ]

        return decoded, features

    def forward(self, images, text_prompts, voting=False):
        """Forward pass through the model"""
        # Extract features from multiple layers
        decoded, all_layer_features = self.extract_features(
            images, text_prompts, voting
        )

        # Process each layer's features with its corresponding classifier
        classifier_outputs = []
        for i in range(self.num_classifiers):
            features = all_layer_features[i]

            # Add shape validation
            if len(features.shape) != 3:
                raise ValueError(
                    f"Expected 3D features [B, T, D], got shape {features.shape}"
                )

            # features = self.encoders[i](features)
            features = self.att_poolings[i](features)
            logits = self.classifiers[i](features)
            classifier_outputs.append(logits)

        if voting:
            return decoded, classifier_outputs
        return classifier_outputs

    def predict(self, images, text_prompts, voting=False):
        """Make predictions with the model"""
        self.eval()
        with torch.no_grad():
            if not voting:
                logits = self.forward(images, text_prompts, voting)
                if not isinstance(logits, list):
                    raise ValueError("Expected logits to be a list")

                # Calculate probabilities and predictions
                probs = [torch.sigmoid(logit.squeeze(-1)) for logit in logits]
                avg_probs = torch.stack(probs).mean(dim=0)
                predictions = (avg_probs > 0.5).float()

                return predictions, avg_probs

            else:
                decoded, logits = self.forward(images, text_prompts, voting)

                if not isinstance(logits, list):
                    raise ValueError("Expected logits to be a list")

                device = logits[0].device

                # Step 1: Classifier votes (weight 1 each)
                classifier_votes = [
                    (torch.sigmoid(logit.squeeze(-1)) > 0.5).float() for logit in logits
                ]

                vote_tensor = torch.stack(
                    classifier_votes
                )  # [num_classifiers, batch_size]

                # Step 2: Process VLM decoded outputs
                vlm_votes = []
                for dec in decoded:
                    d = dec.strip().lower()
                    if d in ["success", "1", "pass"]:
                        vlm_votes.append(1)
                    elif d in ["fail", "0", "failure"]:
                        vlm_votes.append(0)
                    else:
                        vlm_votes.append(None)

                # Step 3: Combine votes with conditional weighting
                predictions = []
                avg_probs = []

                for i, vlm_vote in enumerate(vlm_votes):
                    clf_score = vote_tensor[:, i].sum()  # sum of 3 classifiers

                    if vlm_vote is None:
                        # Only 3 classifiers, threshold is majority (>= 2)
                        total_score = clf_score
                        threshold = 2
                        max_score = 3
                    else:
                        # 3 classifiers + VLM (weight 2), threshold is >= 3 out of 5
                        total_score = clf_score + (2 * vlm_vote)
                        threshold = 3
                        max_score = 5

                    predictions.append(1.0 if total_score >= threshold else 0.0)
                    avg_probs.append(total_score / max_score)

                predictions = torch.tensor(predictions, device=device)
                avg_probs = torch.tensor(avg_probs, device=device)

                return predictions, avg_probs

    def cleanup(self):
        """Remove hooks and clean up resources"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.layer_features.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def save_classifier(self, path="./checkpoints", epoch=None):
        """Save model checkpoints"""
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "num_classifiers": self.num_classifiers,
            "dropout_rate": self.dropout_rate,
        }

        # Save each component
        for i in range(self.num_classifiers):
            checkpoint[f"classifier_{i}"] = self.classifiers[i].state_dict()
            # checkpoint[f'encoder_{i}'] = self.encoders[i].state_dict()
            checkpoint[f"attention_pooling_{i}"] = self.att_poolings[i].state_dict()

        # Add optimizer state if needed for resuming training
        if epoch is not None:
            checkpoint["epoch"] = epoch
            filename = f"components_epoch_{epoch}.pt"
        else:
            filename = "components.pt"

        full_path = os.path.join(path, filename)
        torch.save(checkpoint, full_path)
        print(f"Model saved to {full_path}")

    def load_classifier(self, path, strict=True):
        assert os.path.isfile(path), f"Checkpoint file not found: {path}"

        print(f"Loading FS blocks checkpoint from {path}")

        """Load model checkpoints"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Validate compatibility
        if checkpoint.get("num_classifiers") != self.num_classifiers:
            raise ValueError(
                f"Checkpoint has {checkpoint.get('num_classifiers')} classifiers, "
                f"but model has {self.num_classifiers}"
            )

        # Load each component
        for i in range(self.num_classifiers):
            self.classifiers[i].load_state_dict(
                checkpoint[f"classifier_{i}"], strict=strict
            )
            # self.encoders[i].load_state_dict(checkpoint[f'encoder_{i}'], strict=strict)
            self.att_poolings[i].load_state_dict(
                checkpoint[f"attention_pooling_{i}"], strict=strict
            )

        return checkpoint.get("epoch")


def train_model(model, train_dataset, val_dataset, config):
    """Training function with improved error handling and monitoring"""
    criterion = nn.BCEWithLogitsLoss()

    # Collect trainable parameters
    trainable_params = []
    for i in range(model.num_classifiers):
        trainable_params.extend(model.classifiers[i].parameters())
        # trainable_params.extend(model.encoders[i].parameters())
        trainable_params.extend(model.att_poolings[i].parameters())

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"] * (len(train_dataset) // config["batch_size"]),
        eta_min=config["lr"] * 0.01,
    )

    best_val_acc = 0
    global_step = 0

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        model.train()
        # Keep VLM in eval mode
        model.vlm_model.eval()

        total_loss = 0
        correct = 0
        total = 0
        batch_losses = []

        progress_bar = tqdm.tqdm(
            range(0, len(train_dataset), config["batch_size"]),
            desc=f"Epoch {epoch + 1}",
        )

        for batch_idx, i in enumerate(progress_bar):
            batch_end = min(i + config["batch_size"], len(train_dataset))
            entries = train_dataset[i:batch_end]

            try:
                batch_images = entries["images"]
                batch_texts = [
                    process_input(entries["images"][z], entries["task"][z])
                    for z in range(len(batch_images))
                ]
                batch_labels = torch.tensor(
                    [
                        0 if entry in ("0", "fail", 0) else 1
                        for entry in entries["label"]
                    ],
                    dtype=torch.float32,
                    device=model.device,
                )

                optimizer.zero_grad()

                # Forward pass
                logits = model(batch_images, batch_texts)

                # Calculate loss for each classifier
                if isinstance(logits, list):
                    losses = []
                    for logit in logits:
                        logit = logit.squeeze(-1)
                        loss = criterion(logit, batch_labels)
                        losses.append(loss)

                    # Average loss across classifiers
                    loss = sum(losses) / len(losses)
                else:
                    raise ValueError("Expected logits to be a list")

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Statistics
                batch_loss = loss.item()
                total_loss += batch_loss
                batch_losses.append(batch_loss)

                with torch.no_grad():
                    probs = [torch.sigmoid(logit.squeeze(-1)) for logit in logits]
                    avg_probs = torch.stack(probs).mean(dim=0)
                    predictions = (avg_probs > 0.5).float()

                    batch_correct = (predictions == batch_labels).sum().item()
                    correct += batch_correct
                    total += batch_labels.size(0)

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{batch_loss:.4f}",
                        "acc": f"{correct / total:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                    }
                )

                global_step += 1

                # Validation at intervals
                if global_step % config["validation_step"] == 0:
                    print(f"\n--- Validation at step {global_step} ---")
                    val_acc = validate_model(model, val_dataset, config["batch_size"])
                    print(f"Validation Acc: {val_acc:.4f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        model.save_classifier(
                            path=config["save_path"], epoch=f"best_step_{global_step}"
                        )
                        print(f"New best validation accuracy: {best_val_acc:.4f}")

                    model.train()
                    model.vlm_model.eval()

                # Periodic memory cleanup
                if batch_idx % 100 == 0:
                    model.layer_features.clear()
                    if model.device.type == "cuda":
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n[Error] Batch {batch_idx} failed: {e}")
                continue

        # End of epoch validation
        print(f"\n--- End of Epoch {epoch + 1} ---")
        val_acc = validate_model(model, val_dataset, config["batch_size"])

        avg_loss = total_loss / len(batch_losses) if batch_losses else 0
        train_acc = correct / total if total > 0 else 0

        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}")

        # Save checkpoint
        model.save_classifier(path=config["save_path"], epoch=epoch + 1)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    batch_sizes = [8, 8, 2, 1]

    vlm_model_ids = [
        # "ACIDE/FailSense-Calvin-1p-3b",
        "ACIDE/FailSense-Calvin-2p-3b"
    ]

    for vlm_model_id, batch_size in zip(vlm_model_ids, batch_sizes):
        # gc.collect()

        # Parse model configuration from name
        model_name = vlm_model_id.split("/")[-1]
        parts = model_name.split("-")
        style = "video" if "Video" in parts else "image"
        pov = 1 if "1p" in parts else 2

        # Training configuration
        config = {
            "lr": 1e-4,
            "weight_decay": 0.1,
            "num_epochs": 10,
            "batch_size": batch_size,
            "validation_step": 100000,
            "save_path": f"./{vlm_model_id}",
            "dropout_rate": 0.5,
            "num_classifiers": 3,
        }

        # Initialize model
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        model = FailSense(
            vlm_model_id,
            device=device,
            dropout_rate=config["dropout_rate"],
            num_classifiers=config["num_classifiers"],
        )

        # Optional: Load pretrained FS blocks weights
        # model.load_classifier("path/to/checkpoint.pt")

        # Load datasets
        train_dataset = load_data(
            dataset_name="calvin", style=style, split="train", pov=pov
        )
        # d1 = load_data(dataset_name="calvin", style=style, split="test", pov=pov, num_entry=200)
        d2 = load_data(
            dataset_name="aha", style=style, split="test", pov=pov, num_entry=138
        )
        d3 = load_data(dataset_name="droid", style=style, split="test", pov=pov)

        val_dataset = datasets.concatenate_datasets([d2, d3])

        try:
            # Train model
            best_acc = train_model(model, train_dataset, val_dataset, config)
        finally:
            # Cleanup
            model.cleanup()
            print("Model cleanup completed")

        torch.mps.empty_cache()
