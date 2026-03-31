# I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models

I-FailSense is a **vision-language model (VLM)** for detecting **language-conditioned robotic failures** from visual observations. This repository contains code for training and evaluating I-FailSense models on robotic datasets DROID, Calvin and AHA.

📄 **Paper:** [https://arxiv.org/abs/2509.16072](https://arxiv.org/abs/2509.16072)

**Conference:** ICRA 2026 🎉

---

## 🚀 Installation

Create a Python environment and install dependencies:

```bash
conda create -n ifailsense python=3.10
conda activate ifailsense
pip install -r requirements.txt
```

---

## 🧪 Evaluation

Pre-trained weights are available for evaluation:

* **LoRA weights:** available on [Hugging Face](https://huggingface.co/collections/ACIDE/failsense-3b)
* **FS block weights:** can be downloaded via `wget`

```bash
wget -c https://github.com/clemgris/I-FailSense/releases/tag/models/FS_blocks.zip
```

Run evaluation with:

```bash
python src/evaluate.py \
    --vlm_model_id ACIDE/FailSense-Calvin-1p-3b \
    --fs_id FS/FailSense-Calvin-1p-3b \
    --dataset_name calvin \
    --result_folder results_calvin_1p
```

This will evaluate the model on the **Calvin dataset** and save results in the specified folder.

---

## 🏋️ Training

### Phase 1: Fine-tuning the base VLM with LoRA

```bash
python src/finetune_VLM.py \
    --pov 1 \
    --batch_size 4 \
    --num_epochs 3
```

### Phase 2: Training the FS Blocks

```bash
python src/finetune_FS.py \
    --dataset_name droid \
    --vlm_model_id ACIDE/FailSense-Calvin-1p-3b \
    --batch_size 4 \
    --num_epochs 10
```

## 🔗 References

```
@inproceedings{ifailsense2026,
  title        = {I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models},
  author       = {Clemence Grislain and Hamed Rahimi and Olivier Sigaud and Mohamed Chetouani},
  booktitle    = {Proceedings of the International Conference on Robotics and Automation (ICRA)},
  year         = {2026},
  url          = {https://arxiv.org/abs/2509.16072}
}
```
