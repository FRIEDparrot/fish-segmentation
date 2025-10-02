# Fine-tuning DETR (Segmentation variant) on Fish Dataset

This setup:
1. Downloads the Kaggle fish dataset via `kagglehub`.
2. Builds a dataset that yields (PIL image, COCO-like annotations).
3. Uses `DetrImageProcessor` inside a custom data collator.
4. Loads `DetrForSegmentation` with full pretrained backbone weights.
5. Resizes the classifier head to new number of fish classes.
6. Trains with Hugging Face `Trainer`.

Notes:
- Panoptic fine-tuning ideally uses richer panoptic annotations. Here we approximate each mask as a single “thing” instance (object) with one polygon (a rectangle). This is sufficient to demonstrate end-to-end fine-tuning.
- For multiple fish per image or instance-wise masks, you'd need to adapt dataset logic to find and separate connected components.
- IoU evaluation shown is illustrative (simplistic); production panoptic metrics are more involved.
