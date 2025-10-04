from datasets import (
    DatasetInfo,
    DatasetDict,
    Features,
    Value,
    ClassLabel,
    Image,
    Array2D,
)

from datasets import (
    Split,
    NamedSplit,
    Dataset as HFDataset,
    load_dataset as hf_load_dataset,
)
import torch
from torch.utils.data import random_split
from Fish_Finetune import FishSegmentDataSet, load_dataset # import the dataset to upload
import os

os.chdir(os.path.dirname(__file__))
"""
Firstly, we import our dataset (FishSegmentDataSet) we want to upload to HuggingFace Hub. 
"""

def construct_hf_dataset_from_subset(subset, features, split: NamedSplit):
    def generator():
        for idx in range(len(subset)):
            yield subset[idx]
    return HFDataset.from_generator(generator, split=split, features=features)

def convert_to_hf_dataset(dataset: FishSegmentDataSet) -> DatasetDict:
    """
    Convert our custom FishSegmentDataSet to HuggingFace DatasetDict
    """
    # Define features
    features = Features({
        'image_id': Value('int32'),
        'image': Image(),  # HF Image
        'mask': Array2D(dtype='int64', shape=(590, 445)),  # HF Array2D
        'class_id': Value('int32'),
        'class_name': Value('string'),
    })   # used for converting list of dicts to Dataset

    # Create dataset dict
    data_dict = {
        'train': [],
        'validation': [],
        'test': []
    }

    # Split dataset into train, validation, test (80%, 10%, 10%)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # since we don't want sorted data, we just do a simple split by fixed random seed
    r = 1024
    # load original data from FishSegmentDataSet
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(r)
    )
    # Using generator to avoid memory leaks

    data_dict["train"] = construct_hf_dataset_from_subset(train_set, features, Split.TRAIN)
    data_dict["validation"] = construct_hf_dataset_from_subset(val_set, features, Split.VALIDATION)
    data_dict["test"] = construct_hf_dataset_from_subset(test_set, features, Split.TEST)
    return DatasetDict(data_dict)

def build_datasets():
    dataset = load_dataset(train_size=1.0)[0]  # load the full dataset
    print("converting to hugging face dataset...")
    # convert to DatasetDict
    hf_dataset = convert_to_hf_dataset(dataset)
    hf_dataset.save_to_disk("./fish_dataset_hf")  # save to local disk for backup

def push_to_hub():
    """
    only a dataset dict or dataset can be pushed to hub
    """
    dataset = hf_load_dataset("./fish_dataset_hf")  # load from local disk
    dataset.push_to_hub("FriedParrot/a-large-scale-fish-dataset", private=True)  # push the dataset to hub

if __name__ == "__main__":
    # build_datasets()
    push_to_hub()
