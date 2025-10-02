from Fish_Pretrain import load_fish_dataset, ImageSegmentDataset, FishSegmentDataCollator
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


def main():
    """
    Example usage of ImageSegmentDataset.

    Loads a sample dataset, prints its size, and displays the keys of the first sample.
    """
    img_path = "/mnt/e/kagglehub_cache/datasets/crowww/a-large-scale-fish-dataset/versions/2/Fish_Dataset/Fish_Dataset/Trout/Trout"
    mask_path = "/mnt/e/kagglehub_cache/datasets/crowww/a-large-scale-fish-dataset/versions/2/Fish_Dataset/Fish_Dataset/Trout/Trout GT"
    dataset = ImageSegmentDataset(img_dirs=img_path, mask_dirs=mask_path)
    print(f"Dataset size: {len(dataset)}")

    # ======= Test the dataset loading function =======
    dataset = load_fish_dataset()
    print(dataset[0])

    # Try load some samples
    dl = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dl:
        print(f"Batch size: {len(batch['pixel_values'])}")
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
        print(f"Segmentation mask shape: {batch['segmentation_mask'].shape}")
        print(f"Class names: {batch['class_name']}")
        break  # Just load one batch for testing

    dataset = load_fish_dataset()
    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50-panoptic",
        use_fast=True,
    )
    collator = FishSegmentDataCollator(processor=processor)  # processor

    # Test data collator
    data_loader = DataLoader(dataset, batch_size=3, collate_fn=collator)
    batch = next(iter(data_loader))
    print("Batch keys:", batch.keys())


if __name__ == "__main__":
    main()
