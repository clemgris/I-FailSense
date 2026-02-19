import random
from copy import deepcopy
from typing import Literal, Optional, Union

from datasets import Dataset, load_dataset


def augment_droid_dataset(test_sample):
    augmented_samples = []
    group_size = 3
    n = len(test_sample)

    for i in range(0, n, group_size):
        # original group as list of dicts
        group = test_sample.select(range(i, i + group_size))
        group_dicts = [dict(group[j]) for j in range(len(group))]

        # keep success examples
        augmented_samples.extend(group_dicts)

        # make a fail group
        fail_group = []
        for j in range(group_size):
            sample = deepcopy(group_dicts[j])
            sample["label"] = "fail"
            # shift task: take task from next group (cyclic shift)
            sample["task"] = test_sample[(i + group_size) % n]["task"]
            fail_group.append(sample)

        augmented_samples.extend(fail_group)

    # convert back to Dataset
    augmented_dataset = Dataset.from_list(augmented_samples)
    return augmented_dataset


def load_data(
    dataset_name="calvin",
    style="image",
    pov=1,
    split="train",
    num_entry: Optional[Union[int, Literal["full"]]] = "full",
    seed=42,
):
    print(
        f"Loading dataset: {dataset_name}, style: {style}, pov: {pov}, split: {split}, num_entry: {num_entry}, seed: {seed}"
    )

    AVAILABLE_DATASETS = {
        "calvin": ["test", "train"],
        "droid": ["test", "train"],
        "aha": ["test", "train"],
    }

    if dataset_name not in AVAILABLE_DATASETS:
        available = ", ".join(AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Available: {available}"
        )

        # Validate split
    if split not in AVAILABLE_DATASETS[dataset_name]:
        available_splits = ", ".join(AVAILABLE_DATASETS[dataset_name])
        raise ValueError(
            f"Split '{split}' not available for dataset '{dataset_name}'. Available: {available_splits}"
        )

        # Validate type
    if style not in ["image", "video"]:
        raise ValueError(f"Type '{style}' not supported. Must be 'image' or 'video'.")

        # Validate pov
    if pov not in [1, 2, 3]:
        raise ValueError(f"POV '{pov}' not supported. Must be 1, 2 or 3.")

    print(f"Validation passed. Processing dataset: {dataset_name}")

    if dataset_name == "calvin":
        print(f"Processing Calvin dataset with style: {style}")
        if style == "image":
            print(f"Loading Calvin image dataset with POV: {pov}")
            if pov == 1:
                if split == "train":
                    print("Loading ACIDE/AHA-Calvin-1p train split")
                    dataset = load_dataset("ACIDE/AHA-Calvin-1p", split=split)
                else:
                    print(
                        "Loading ACIDE/AHA-Calvin-1p validation split and creating test split"
                    )
                    dataset = load_dataset("ACIDE/AHA-Calvin-1p", split="validation")
                    dataset = dataset.train_test_split(test_size=0.1, seed=seed)
                    dataset = dataset[split]

            elif pov == 2:
                if split == "train":
                    print("Loading ACIDE/AHA-Calvin-2p train split")
                    dataset = load_dataset("ACIDE/AHA-Calvin-2p", split=split)
                else:
                    print(
                        "Loading ACIDE/AHA-Calvin-2p validation split and creating test split"
                    )
                    dataset = load_dataset("ACIDE/AHA-Calvin-2p", split="validation")
                    dataset = dataset.train_test_split(test_size=0.1, seed=seed)
                    dataset = dataset[split]

            else:
                raise ValueError(f"POV '{pov}' is not supported for {dataset_name}")

            print("Renaming columns: image -> images, success -> label")
            dataset = dataset.rename_column("image", "images")
            dataset = dataset.rename_column("success", "label")

        else:
            print(f"Loading Calvin video dataset with POV: {pov}")
            if pov == 1:
                if split == "train":
                    print("Loading ACIDE/AHA-Calvin train split")
                    dataset = load_dataset("ACIDE/AHA-Calvin", split=split)
                else:
                    print(
                        "Loading ACIDE/AHA-Calvin validation split and creating test split"
                    )
                    dataset = load_dataset("ACIDE/AHA-Calvin", split="validation")
                    dataset = dataset.train_test_split(test_size=0.1, seed=seed)
                    dataset = dataset[split]

                print("Creating samples from images_1 and images_2")
                samples = []
                for i, item in enumerate(dataset):
                    if i % 100 == 0:
                        print(f"Processing item {i}/{len(dataset)}")
                    samples.append(
                        {
                            "images": item["images_1"],
                            "task": item["task"],
                            "label": item["success"],
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_2"],
                            "task": item["task"],
                            "label": item["success"],
                        }
                    )
                print(
                    f"Created {len(samples)} samples from {len(dataset)} original items"
                )
                dataset = Dataset.from_list(samples)
                del samples

            elif pov == 2:
                if split == "train":
                    print("Loading ACIDE/AHA-Calvin train split")
                    dataset = load_dataset("ACIDE/AHA-Calvin", split=split)
                else:
                    print(
                        "Loading ACIDE/AHA-Calvin validation split and creating test split"
                    )
                    dataset = load_dataset("ACIDE/AHA-Calvin", split="validation")
                    dataset = dataset.train_test_split(test_size=0.1, seed=seed)
                    dataset = dataset[split]

                print("Creating samples by concatenating images_1 and images_2")
                samples = []
                for i, item in enumerate(dataset):
                    if i % 100 == 0:
                        print(f"Processing item {i}/{len(dataset)}")
                    samples.append(
                        {
                            "images": item["images_1"] + item["images_2"],
                            "task": item["task"],
                            "label": item["success"],
                        }
                    )

                    samples.append(
                        {
                            "images": item["images_2"] + item["images_1"],
                            "task": item["task"],
                            "label": item["success"],
                        }
                    )

                print(
                    f"Created {len(samples)} samples from {len(dataset)} original items"
                )
                dataset = Dataset.from_list(samples)
                del samples
            else:
                raise ValueError(f"POV '{pov}' is not supported for {dataset_name}")

    elif dataset_name == "droid":
        print(f"Processing DROID dataset with style: {style}")
        if style == "image":
            print(f"Loading DROID image dataset with POV: {pov}")
            if pov == 1:
                if split == "test":
                    print("Loading ACIDE/DROID_1p_bench")
                    dataset = load_dataset("ACIDE/DROID_1p_bench", split="train")
                else:
                    print("Loading ACIDE/DROID_1p_1k")
                    dataset = load_dataset("ACIDE/DROID_1p_1k", split="train")
            elif pov == 2:
                if split == "test":
                    print("Loading ACIDE/DROID_2p_bench")
                    dataset = load_dataset("ACIDE/DROID_2p_bench", split="train")
                else:
                    print("Loading ACIDE/DROID_2p_1k")
                    dataset = load_dataset("ACIDE/DROID_2p_1k", split="train")
            else:
                # TBD (hamed) :  function for creating 3pov images from "ACIDE/DROID_Bench"
                raise ValueError(
                    f"POV '{pov}' is not supported for {dataset_name} in {style} style at the moment,"
                    f" but since data is there, it is possible to produce it."
                )
            print("Renaming column: success -> label")
            dataset = dataset.rename_column("success", "label")

        else:
            print("Loading ACIDE/DROID_Bench for video processing")

            if split == "test":
                dataset = load_dataset("ACIDE/DROID_Bench", split="train")
            else:
                dataset = load_dataset("ACIDE/DROID_bench_1k", split="train")
            if pov == 1:
                print("Creating samples from images_1, images_2, and images_3")
                samples = []
                for i, item in enumerate(dataset):
                    if i % 100 == 0:
                        print(f"Processing item {i}/{len(dataset)}")
                    samples.append(
                        {
                            "images": item["images_1"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_2"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_3"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                print(
                    f"Created {len(samples)} samples from {len(dataset)} original items"
                )
                dataset = Dataset.from_list(samples)
                del samples

            elif pov == 2:
                print(
                    "Creating samples from pairwise combinations of images_1, images_2, and images_3"
                )
                samples = []
                for i, item in enumerate(dataset):
                    if i % 100 == 0:
                        print(f"Processing item {i}/{len(dataset)}")
                    samples.append(
                        {
                            "images": item["images_1"] + item["images_2"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_2"] + item["images_1"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_1"] + item["images_3"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_3"] + item["images_1"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_2"] + item["images_3"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_3"] + item["images_2"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                print(
                    f"Created {len(samples)} samples from {len(dataset)} original items"
                )
                dataset = Dataset.from_list(samples)
                del samples

            else:
                print(
                    "Creating samples from all permutations of images_1, images_2, and images_3"
                )
                samples = []
                for i, item in enumerate(dataset):
                    if i % 100 == 0:
                        print(f"Processing item {i}/{len(dataset)}")
                    samples.append(
                        {
                            "images": item["images_1"]
                            + item["images_2"]
                            + item["images_3"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_1"]
                            + item["images_3"]
                            + item["images_2"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_2"]
                            + item["images_1"]
                            + item["images_3"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_2"]
                            + item["images_3"]
                            + item["images_1"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_3"]
                            + item["images_1"]
                            + item["images_2"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                    samples.append(
                        {
                            "images": item["images_3"]
                            + item["images_2"]
                            + item["images_3"],
                            "task": item["task"],
                            "label": "success",
                        }
                    )
                print(
                    f"Created {len(samples)} samples from {len(dataset)} original items"
                )
                dataset = Dataset.from_list(samples)
                del samples

    else:
        print(f"Processing AHA dataset with style: {style}")
        if style == "image":
            if split == "train":
                print(f"Loading AHA training dataset with POV: {pov}")
                if pov == 1:
                    print("Loading ACIDE/AHA_Dataset_2e_1p")
                    dataset = load_dataset("ACIDE/AHA_Dataset_2e_1p", split="train")
                elif pov == 2:
                    print("Loading ACIDE/AHA_Dataset_2e_2p")
                    dataset = load_dataset("ACIDE/AHA_Dataset_2e_2p", split="train")
                else:
                    # TBD (hamed) :  function for creating 3pov images from "ACIDE/DROID_Bench"
                    raise ValueError(
                        f"POV '{pov}' is not supported for {dataset_name} in {style} style at the moment,"
                        f" but since data is there, it is possible to produce it."
                    )
                print("Renaming columns: image -> images, success -> label")
                dataset = dataset.rename_column("image", "images")
                dataset = dataset.rename_column("success", "label")
            else:
                print(f"Loading AHA test dataset with POV: {pov}")
                if pov == 1:
                    print("Loading ACIDE/AHA_bench_1p")
                    dataset = load_dataset("ACIDE/AHA_bench_1p", split="train")
                elif pov == 2:
                    print("Loading ACIDE/AHA_bench_2p")
                    dataset = load_dataset("ACIDE/AHA_bench_2p", split="train")
                else:
                    # TBD (hamed) :  function for creating 3pov images from "ACIDE/DROID_Bench"
                    raise ValueError(
                        f"POV '{pov}' is not supported for {dataset_name} in {style} style at the moment,"
                        f" but since data is there, it is possible to produce it."
                    )
                print("Renaming columns: image -> images, success -> label")
                dataset = dataset.rename_column("image", "images")
                dataset = dataset.rename_column("success", "label")

        else:
            if split == "train":
                print("Loading ACIDE/AHA_Dataset_2e for video processing")
                dataset = load_dataset("ACIDE/AHA_Dataset_2e", split="train")
                if pov == 1:
                    print(
                        "Creating samples from images_front, images_wrist, and images_overhead"
                    )
                    samples = []
                    for i, item in enumerate(dataset):
                        if i % 100 == 0:
                            print(f"Processing item {i}/{len(dataset)}")
                        samples.append(
                            {
                                "images": item["images_front"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_wrist"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_overhead"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                    print(
                        f"Created {len(samples)} samples from {len(dataset)} original items"
                    )
                    dataset = Dataset.from_list(samples)
                    del samples
                elif pov == 2:
                    print("Creating samples from pairwise combinations of camera views")
                    samples = []
                    for i, item in enumerate(dataset):
                        if i % 100 == 0:
                            print(f"Processing item {i}/{len(dataset)}")
                        samples.append(
                            {
                                "images": item["images_front"] + item["images_wrist"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_wrist"] + item["images_front"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_overhead"]
                                + item["images_front"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_front"]
                                + item["images_overhead"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_wrist"]
                                + item["images_overhead"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_overhead"]
                                + item["images_wrist"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                    print(
                        f"Created {len(samples)} samples from {len(dataset)} original items"
                    )
                    dataset = Dataset.from_list(samples)
                    del samples
                else:
                    print("Creating samples from all permutations of camera views")
                    samples = []
                    for i, item in enumerate(dataset):
                        if i % 100 == 0:
                            print(f"Processing item {i}/{len(dataset)}")
                        samples.append(
                            {
                                "images": item["images_front"]
                                + item["images_wrist"]
                                + item["images_overhead"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_front"]
                                + item["images_overhead"]
                                + item["images_wrist"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_wrist"]
                                + item["images_front"]
                                + item["images_overhead"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_wrist"]
                                + item["images_overhead"]
                                + item["images_front"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_overhead"]
                                + item["images_wrist"]
                                + item["images_front"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_overhead"]
                                + item["images_front"]
                                + item["images_wrist"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                    print(
                        f"Created {len(samples)} samples from {len(dataset)} original items"
                    )
                    dataset = Dataset.from_list(samples)
                    del samples

            else:
                print("Loading ACIDE/AHA_bench for video processing")
                dataset = load_dataset("ACIDE/AHA_bench", split="train")
                if pov == 1:
                    print("Creating samples from images_1 and images_2")
                    samples = []
                    for i, item in enumerate(dataset):
                        if i % 100 == 0:
                            print(f"Processing item {i}/{len(dataset)}")
                        samples.append(
                            {
                                "images": item["images_1"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_2"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                    print(
                        f"Created {len(samples)} samples from {len(dataset)} original items"
                    )
                    dataset = Dataset.from_list(samples)
                    del samples
                elif pov == 2:
                    print("Creating samples from combinations of images_1 and images_2")
                    samples = []
                    for i, item in enumerate(dataset):
                        if i % 100 == 0:
                            print(f"Processing item {i}/{len(dataset)}")
                        samples.append(
                            {
                                "images": item["images_1"] + item["images_2"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                        samples.append(
                            {
                                "images": item["images_2"] + item["images_1"],
                                "task": item["task"],
                                "label": "fail",
                            }
                        )
                    print(
                        f"Created {len(samples)} samples from {len(dataset)} original items"
                    )
                    dataset = Dataset.from_list(samples)
                    del samples
                else:
                    raise ValueError(
                        f"POV '{pov}' is not supported for {dataset_name} in {style} style at the moment,"
                        f" but since data is there, it is possible to produce it."
                    )

    dataset_length = len(dataset)
    print(f"Dataset loaded with {dataset_length} samples")

    if num_entry == "full" or num_entry is None:
        print("Returning full dataset")
        return dataset
    elif isinstance(num_entry, int):
        print(f"Sampling {num_entry} entries from {dataset_length} total samples")

        # Set random seed for reproducibility
        random.seed(seed)

        # Randomly select indices
        sampled_indices = random.sample(range(dataset_length), num_entry)
        print(
            f"Selected indices: {sampled_indices[:10]}{'...' if len(sampled_indices) > 10 else ''}"
        )

        # Use the select method to create a new dataset
        sampled_dataset = dataset.select(sampled_indices)
        print(f"Sampled dataset created with {len(sampled_dataset)} samples")

        return sampled_dataset
    else:
        raise ValueError(
            f"Invalid sample_size: {num_entry}. Must be 'full' or integer."
        )


# Example usage
if __name__ == "__main__":
    # Load full training dataset with default parameters (image, POV 1)
    calvin_train = load_data(dataset_name="calvin", split="train", num_entry="full")
    print(f"Calvin train dataset size: {len(calvin_train)}")
    print("###############################")
    # Load random sample of test dataset with video type and POV 2
    calvin_test_sample = load_data(
        dataset_name="calvin", split="test", num_entry=100, style="video", pov=2
    )
    print(f"Calvin test sample size: {len(calvin_test_sample)}")
    print("###############################")
    # Load droid test dataset with image type and POV 1
    droid_test = load_data(dataset_name="droid", split="test", style="image", pov=1)
    print(f"Droid test dataset size: {len(droid_test)}")
    print("###############################")
    # Load AHA training dataset with video type and POV 1
    aha_train = load_data(
        dataset_name="aha", split="train", num_entry="full", style="video", pov=1
    )
    print(f"AHA train dataset size: {len(aha_train)}")
