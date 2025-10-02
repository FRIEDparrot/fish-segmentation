import numpy as np
from torch.utils.data import Dataset, random_split
from PIL import Image
import torch
from typing import Optional, Dict, Union
import kagglehub
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List
import torch.nn.functional as F


# region Utility functions

def _compute_bbox_from_mask(mask_np: np.ndarray):
    """
    Compute (x_min, y_min, width, height) in absolute pixel coords from a binary mask.
    Returns None if mask is empty.
    """
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return [int(x_min), int(y_min), int(width), int(height)]


def resize_mask(mask, size=(800, 1060)) -> torch.Tensor:
    """
    Resize segmentation mask to given size using nearest neighbor interpolation.

    Args:
        mask (torch.Tensor or PIL.Image): Input mask.
            - If torch.Tensor: shape (H, W) or (1, H, W).
            - If PIL.Image: will be converted to tensor.
        size (tuple): target size (height, width).

    Returns:
        torch.Tensor: resized mask, shape (H_new, W_new), dtype long
    """
    if isinstance(mask, Image.Image):
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    if mask.ndim == 2:  # (H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)
    elif mask.ndim == 3 and mask.shape[0] == 1:  # (1,H,W)
        mask = mask.unsqueeze(0)  # -> (1,1,H,W)
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    # nearest neighbor interpolation for masks
    mask_resized = F.interpolate(mask.float(), size=size, mode="nearest")
    mask_bin = (mask_resized > 0).long()  # Binarize
    return mask_bin.squeeze()  # -> (H_new, W_new)


#endregion

#region Dataset and DataCollator
class ImageSegmentDataset(Dataset):
    """
    PyTorch Dataset for image segmentation tasks.

    This dataset expects images and their corresponding masks to have the same filenames
    and be stored in separate directories. It supports multiple directories for images and masks,
    optional labels, and class names for each directory.

    Args:
        img_dirs (Union[str, List[str]]): Path(s) to image directory/directories.
        mask_dirs (Union[str, List[str]]): Path(s) to mask directory/directories.
        label (Optional[Union[int, List[int]]], optional): Optional label(s) for each directory.
        class_names (Optional[Union[str, List[str]]], optional): Optional class name(s) for each directory.
        file_pattern (Optional[str], optional): Optional glob pattern to filter files (e.g., "*.png").
        name_strict (Optional[bool], optional): If True, enforces strict filename matching between images and masks.

    Raises:
        ValueError: If the number of image and mask directories, labels, or class names do not match,
                    or if image and mask filenames do not match when name_strict is True.

    Example:
        dataset = ImageSegmentDataset(
            img_dirs=["images/fish", "images/shark"],
            mask_dirs=["masks/fish", "masks/shark"],
            label=[0, 1],
            class_names=["fish", "shark"]
        )

    Returns:
        only loads raw tensors (no processing)

        keys : "image_id", "pixel_values", "segmentation_mask", optional "label", "class_name"
    """

    def __init__(self,
                 img_dirs: Union[str, List[str]],
                 mask_dirs: Union[str, List[str]],
                 label: Optional[Union[int, List[int]]] = None,
                 class_names: Optional[Union[str, List[str]]] = None,
                 file_pattern: Optional[str] = None,
                 name_strict: Optional[bool] = True):
        super().__init__()
        # Normalize inputs to lists
        self.img_dirs: List[str] = [img_dirs] if isinstance(img_dirs, str) else img_dirs
        self.mask_dirs: List[str] = [mask_dirs] if isinstance(mask_dirs, str) else mask_dirs
        if len(self.img_dirs) != len(self.mask_dirs):
            raise ValueError("Number of image and mask directories must match")

        # Handle labels and class names
        if label is not None:
            self.labels = [label] if isinstance(label, int) else label
            if len(self.labels) != len(self.img_dirs):
                raise ValueError("Number of labels must match number of directories")
        else:
            self.labels = [None] * len(self.img_dirs)

        if class_names is not None:
            self.class_names = [class_names] if isinstance(class_names, str) else class_names
            if len(self.class_names) != len(self.img_dirs):
                raise ValueError("Number of class names must match number of directories")
        else:
            self.class_names = [None] * len(self.img_dirs)

        self.file_pattern = file_pattern
        self._build_index(name_strict=name_strict)

    def _build_index(self, name_strict=True):
        """
        Builds the index mapping between image and mask files for all directories.

        Args:
            name_strict (bool): If True, enforces strict filename matching between images and masks.

        Raises:
            ValueError: If the number of images and masks do not match, or filenames do not match when name_strict is True.
        """
        self.index: list[Dict] = []
        for dir_idx, (img_dir, mask_dir, label, class_name) in enumerate(
                zip(self.img_dirs, self.mask_dirs, self.labels, self.class_names)
        ):
            if self.file_pattern:
                sample_images = sorted(Path(img_dir).glob(self.file_pattern))
                sample_masks = sorted(Path(mask_dir).glob(self.file_pattern))
                sample_images = [f.name for f in sample_images]
                sample_masks = [f.name for f in sample_masks]
            else:
                sample_images = sorted(os.listdir(img_dir))
                sample_masks = sorted(os.listdir(mask_dir))

            if len(sample_images) != len(sample_masks):
                raise ValueError(f"Mismatch in {img_dir}: {len(sample_images)} images, {len(sample_masks)} masks")
            if name_strict and sample_images != sample_masks:
                raise ValueError(f"Image and mask file lists do not match in directory pair {img_dir}, {mask_dir}")

            for img_file, mask_file in zip(sample_images, sample_masks):
                self.index.append({
                    "img_path": os.path.join(img_dir, img_file),
                    "mask_path": os.path.join(mask_dir, mask_file),
                    "label": label,
                    "class_name": class_name
                })

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        Retrieve the list of fish classes from the dataset.

        This function downloads the fish dataset using `kagglehub`, navigates to the dataset directory,
        and returns a list of class names based on the subdirectories present.

        Returns:
            list[str]: A list of fish class names, where each class corresponds to a subdirectory in the dataset.
        """
        item = self.index[idx]
        image = Image.open(item["img_path"]).convert("RGB")  # RGB input
        mask = Image.open(item["mask_path"]).convert("L")  # convert mask to grayscale)
        mask_np = np.array(mask, dtype=np.int64)

        pixel_values = torch.tensor(np.array(image)).permute(2, 0, 1)  # add batch dimension
        result = {
            "image_id": idx,
            "pixel_values": pixel_values,
            "segmentation_mask": torch.tensor(mask_np, dtype=torch.long),  # convert mask to tensor
        }

        # Add optional label information
        if item["label"] is not None:
            result["label"] = torch.tensor(item["label"], dtype=torch.long)
        if item["class_name"] is not None:
            result["class_name"] = item["class_name"]
        return result


@dataclass
class FishSegmentDataCollator:
    processor: any  # Replace 'any' with the actual type of your processor

    def __call__(self, batch):
        """
        Features is dataset,
        """
        images = torch.stack([feature["pixel_values"] for feature in batch])  #
        seg_masks = torch.stack([resize_mask(feature["segmentation_mask"], size=(800, 1060)) for feature in batch])

        # resize masks to  [800, 1060]
        class_names = [feature.get("class_name", None) for feature in batch]  # class name
        class_ids = [feature.get("label", None) for feature in batch]  # class id

        # including : pixel_values, pixel_mask  (returns numpy arrays)
        encodings = self.processor(images)

        meta = []
        bboxes = []
        for mask, cls_name in zip(seg_masks, class_names):  # masks = list of numpy or torch 2D arrays
            bbox = _compute_bbox_from_mask(mask.cpu().numpy())
            if bbox is None:
                continue
            bboxes.append(bbox)
            area = bbox[2] * bbox[3]
            x, y, w, h = bbox
            # 4 vertices of the rectangle
            segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]

            meta.append({
                "class_name": cls_name,
                "area": torch.tensor([area]),
                "segmentation": segmentation,
            })
        result = dict(encodings)  # expand storage

        # return tensor for training parameters
        result["pixel_values"] = torch.tensor(np.stack(encodings["pixel_values"]), dtype=torch.float)
        result["pixel_mask"] = torch.tensor(np.stack(encodings["pixel_mask"]), dtype=torch.long)  # attention mask
        result["segmentation_mask"] = seg_masks  # caution: not use "pixel_mask"
        result["class_ids"] = torch.stack(class_ids, dim=0)
        result["bboxes"] = torch.tensor(bboxes, dtype=torch.float) if bboxes else torch.empty((0, 4), dtype=torch.float)

        result["meta"] = meta  # meta data list
        return result


# endregion

def get_fish_classes():
    dataset_url = "crowww/a-large-scale-fish-dataset"
    base_path = os.path.join(kagglehub.dataset_download(dataset_url), "Fish_Dataset", "Fish_Dataset")
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]


def load_fish_dataset():
    """
    Loads the fish segmentation dataset from Kaggle using kagglehub.

    Returns:
        ImageSegmentDataset: Dataset containing fish images and masks, organized by class.
    """
    dataset_url = "crowww/a-large-scale-fish-dataset"
    base_path = os.path.join(kagglehub.dataset_download(dataset_url), "Fish_Dataset", "Fish_Dataset")
    class_names = get_fish_classes()  # Add "N/A" for background class
    class_ids = list(range(len(class_names)))

    img_dirs = [os.path.join(base_path, cls, cls) for cls in class_names]
    mask_dirs = [os.path.join(base_path, cls, cls + " GT") for cls in class_names]
    dataset = ImageSegmentDataset(
        img_dirs=img_dirs,
        mask_dirs=mask_dirs,
        class_names=class_names,
        label=class_ids,
        name_strict=True
    )
    return dataset


def load_fish_datasets_all(train_size=0.7, test_size=0.2, random_seed=42):
    """
    Splits the fish segmentation dataset into training, validation, and test sets.
    """
    assert train_size + test_size < 1.0, "train_size and test_size must sum to less than 1.0"
    dataset = load_fish_dataset()
    train_len = int(train_size * len(dataset))
    test_len = int(test_size * len(dataset))
    val_len = len(dataset) - train_len - test_len

    g = torch.Generator().manual_seed(random_seed)
    train_set, test_set, val_set = random_split(
        dataset, [train_len, test_len, val_len],
        generator=g
    )
    return train_set, test_set, val_set
