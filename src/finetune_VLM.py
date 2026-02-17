import argparse
import os

import torch
from load_dataset import load_data
from peft import LoraConfig, get_peft_model
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)


def train_paligemma(
    model_id="google/paligemma2-3b-mix-224",
    ds=load_data(dataset_name="calvin"),
    output_dir="Calvin-3b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=1e-5,
    logging_steps=500,
    save_steps=500,
    eval_steps=1000,
    save_total_limit=1,
    warmup_ratio=0.1,
    report_to=["tensorboard"],
):
    # Setup device
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processor and model
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )

    EOS_TOKEN = processor.tokenizer.eos_token
    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

    # Apply LoRA
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_cfg)
    model = model.to(device)

    # Freeze parts of the model
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    # Define collate function
    def collate_fn(examples):
        texts = [
            "<image>" * example["images"] + " evaluate en " + example["task"]
            for example in examples
        ]
        labels = [
            "success" + EOS_TOKEN
            if example["label"] in ["1", "success"]
            else "fail" + EOS_TOKEN
            for example in examples
        ]
        images = [example["images"] for example in examples]
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
        )
        tokens = tokens.to(model.dtype).to(device)
        return tokens

    # Set up training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta2=0.999,
        logging_steps=logging_steps,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        push_to_hub=True,
        bf16=True,
        dataloader_pin_memory=False,
        local_rank=local_rank,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        report_to=report_to,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=collate_fn,
        args=args,
    )

    # Train
    trainer.train()

    # Push to hub
    processor.push_to_hub(output_dir)
    model.push_to_hub(output_dir)


def main(args):
    # Load dataset
    dataset = load_data(
        dataset_name="calvin", style="image", split="train", pov=args.pov
    )
    train_paligemma(ds=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune PaliGemma on I-FailSense datasets"
    )
    parser.add_argument(
        "--pov",
        type=int,
        choices=[1, 2],
        default=2,
        help="Point of view for the dataset (1 or 2)",
    )

    args = parser.parse_args()

    main(args)
