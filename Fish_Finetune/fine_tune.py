import os
import kagglehub
import torch
from transformers import DetrConfig, DetrForSegmentation, DetrImageProcessor, BatchFeature
from transformers import Trainer, TrainingArguments
from transformers.image_utils import AnnotationFormat, AnnotationType
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Union, List, Any
import numpy as np
import warnings
from Fish_Pretrain.dataset_loading import compute_bbox_from_mask, resize_mask, get_fish_classes
from Fish_Pretrain.utils import backup_model_to_hub

# Close all warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


class FishSegmentDataSet(Dataset):
    def __init__(self,
                 img_dirs: Union[str, List[str]],
                 mask_dirs: Union[str, List[str]]):
        super().__init__()
        self.img_dirs: List[str] = [img_dirs] if isinstance(img_dirs, str) else img_dirs
        self.mask_dirs: List[str] = [mask_dirs] if isinstance(mask_dirs, str) else mask_dirs
        if len(self.img_dirs) != len(self.mask_dirs):
            raise ValueError("Number of image and mask directories must match")

        self.items = []
        for idx, (img_dir, mask_dir) in enumerate(zip(self.img_dirs, self.mask_dirs)):
            img_paths = sorted(os.listdir(img_dir))
            mask_paths = sorted(os.listdir(mask_dir))
            class_name = os.path.basename(img_dir)  # assuming img_dir is like .../class_name
            if len(img_paths) != len(mask_paths):
                raise ValueError(f"Number of images and masks must match in {img_dir} and {mask_dir}")
            for img_path, mask_path in zip(img_paths, mask_paths):
                img_id = os.path.splitext(img_path)[0]
                mask_id = os.path.splitext(mask_path)[0]
                class_id = get_fish_classes().index(class_name)

                if img_id != mask_id:
                    raise ValueError(f"Image and mask file names must match: {img_id} vs {mask_id}")
                self.items.append(dict(img_path=img_path,
                                       mask_path=mask_path,
                                       class_name=class_name,
                                       class_id=class_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item_id):
        record = self.items[item_id]
        img = Image.open(os.path.join(self.img_dirs[record['class_id']], record['img_path']))
        mask = Image.open(os.path.join(self.mask_dirs[record['class_id']], record['mask_path'])).convert("L")
        # here we also return image_id for tracking
        return dict(image_id=item_id,
                    image=img,
                    mask=torch.tensor(np.array(mask)).long(),
                    class_id=record['class_id'],
                    class_name=record['class_name'])


class FishCollator:
    def __init__(self, processor: DetrImageProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> BatchFeature:
        images = [f["image"] for f in features]
        annotations = []

        # we may have multiple objects in one image, but here in dataset we use only one object per image
        for f in features:
            # prepare annotations from masks
            bbox = compute_bbox_from_mask(np.array(f["mask"]))
            area = float(bbox[2] * bbox[3])  # absolute pixel area
            ann: AnnotationType = {
                "image_id": torch.tensor(f["image_id"]),
                "annotations": [
                    {
                        "image_id": torch.tensor(f["image_id"]),  # not set to class_id, It breaks the matching logic inside the loss computation (Hungarian matching).
                        "category_id": torch.tensor(f["class_id"]),
                        "bbox": torch.tensor(bbox),  # only gives bbox and area
                        "area": torch.tensor(area),
                        # for COCO polygon format,
                        # "segmentation": [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3]]],
                        "iscrowd": torch.tensor([0]),
                    }
                ]
            }
            annotations.append(ann)
        # it is a dict, but here we use batch feature for convenience
        out = self.processor(images,
                             annotations=annotations,
                             return_segmentation_masks=False,  # no mask info
                             format=AnnotationFormat.COCO_DETECTION,
                             return_tensors="pt")
        # since out now have no masks, we need to manually inject them back
        masks = [f["mask"] for f in features]

        for i, mask in enumerate(masks):
            # resize mask to the size of out['labels'][i]['size']
            target_size = tuple(out['labels'][i]['size'])
            mask_resize = resize_mask(mask, size=target_size)
            out['labels'][i]["masks"] = torch.tensor(mask_resize, dtype=torch.float32).unsqueeze(0)  # add channel dim
        return out


def load_dataset(train_size: float = 0.8, seed: int = 42):
    """
    Loads the fish segmentation dataset and splits it into training and validation sets.

    Args:
        train_size (float): Proportion of the dataset to use for training (between 0 and 1).
        seed (int): Random seed for reproducibility.

    Returns:
        train_dataset: Training subset of FishSegmentDataSet.
        val_dataset: Validation subset of FishSegmentDataSet (None if train_size=1).
    """
    dataset_url = "crowww/a-large-scale-fish-dataset"
    base_path = os.path.join(kagglehub.dataset_download(dataset_url), "Fish_Dataset", "Fish_Dataset")
    class_names = get_fish_classes()
    img_dirs = [os.path.join(base_path, cls, cls) for cls in class_names]
    mask_dirs = [os.path.join(base_path, cls, cls + " GT") for cls in class_names]
    dataset = FishSegmentDataSet(img_dirs=img_dirs, mask_dirs=mask_dirs)

    if train_size >= 1.0:
        return dataset, None
    if train_size <= 0.0:
        return None, dataset
    train_len = int(len(dataset) * train_size)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
    return train_dataset, val_dataset


def train_model():
    # just use pretrained DETR model and fine-tune it
    base_model_name = "facebook/detr-resnet-50-panoptic"
    config = DetrConfig.from_pretrained(
        base_model_name,  # **this is confi from pretrained DETR model**
        num_labels=9  # use number of Fish species (no need to +1 for background in DETR segmentation)
    )
    config.id2label = {i: cls for i, cls in enumerate(get_fish_classes())}
    config.label2id = {v: k for k, v in config.id2label.items()}
    config.model_name="Fish_Segmentation_Model_Fine_Tuning_DETR"

    model = DetrForSegmentation.from_pretrained(
        base_model_name,  # number of fish species
        ignore_mismatched_sizes=True,
        config=config,
    )
    processor = DetrImageProcessor.from_pretrained(base_model_name)

    train_dataset, val_dataset = load_dataset(train_size=0.8, seed=42)
    collator = FishCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir="./models",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,           # 15 - 20
        remove_unused_columns=False,   # This must be specified
        eval_strategy="steps",
        learning_rate=5e-5,   # for a larger model, consider using a smaller learning rate is better.
        weight_decay=0.025,
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        logging_dir="./logs",
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    """
    What we expect : 
    except the background class[10], we have only 1 bbox per image, and then masks should also be 
    predicted correctly.  
    """
    trainer.save_model('./models/final_model')
    processor.save_pretrained('./models/final_model')
    config.save_pretrained('./models/final_model')

    repo_id = "FriedParrot/fish-segmentation-simple"
    # push model to hub
    backup_model_to_hub(repo_id, config, model, processor)
    print("Model training and saving completed.")



if __name__ == "__main__":
    train_model()