from transformers import AutoProcessor
from Fish_Pretrain import (
    FishSegmentationModel, FishSegmentModelConfig,
    load_fish_datasets_all, FishSegmentDataCollator, get_fish_classes,
    FishSegmentationModelOutput
)
from torch.utils.data import DataLoader
import os
import torch
from Fish_Pretrain.utils import visualize_sample_comparison
from torchinfo import summary

#region Model Loading and Summary
def load_models():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    model_path = "./models/final_model"
    config = FishSegmentModelConfig.from_pretrained(model_path, local_files_only=True)
    model = FishSegmentationModel.from_pretrained(model_path, config=config, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    return model, config, processor

def print_model_summary(model):
    batch_size = 1
    channels = 3
    height = 590
    width = 445
    pixel_values = torch.randn(batch_size, channels, height, width)
    pixel_mask = torch.ones(batch_size, height, width)
    summary(model, input_data={"pixel_values": pixel_values, "pixel_mask": pixel_mask})
#endregion

# -------------------------------
# Evaluation Loop
# -------------------------------

def test_fish_segmentation_model(
    model: FishSegmentationModel, processor, batch_size: int = 4, max_vis: int = 2, save_dir: str = None
):
    """Evaluate on test set, report classification acc, and visualize a few samples."""
    _, _, test_set = load_fish_datasets_all()
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                        num_workers=2, collate_fn=FishSegmentDataCollator(processor))
    classes = get_fish_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        showed_num = 0
        correct = 0 # number of correct classification
        for b_idx, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out: FishSegmentationModelOutput = model(**batch)
            labels, preds = batch.get("class_ids"), out.class_logits.argmax(-1)
            correct += (preds == labels).sum().item()
            if showed_num > max_vis:
                print("evaluated batch : ", b_idx, "/" , len(loader))
                continue

            # visualize only up to max_vis samples
            for i, img in enumerate(batch["pixel_values"][:max_vis].cpu()):
                if showed_num > max_vis:
                    continue
                showed_num += 1
                gt_mask = (batch.get("segmentation_mask")[i] > 0).long() if "segmentation_mask" in batch else None
                pred_mask = (out.seg_logits[i].argmax(0) > 0).long()
                gt_name = classes[labels[i].item()] if labels is not None else "Unknown"
                pred_name = classes[preds[i].item()]

                visualize_sample_comparison(
                    image=img.cpu(),
                    gt_mask=gt_mask.to("cpu") if gt_mask is not None else None,
                    pred_mask=pred_mask.to("cpu") if pred_mask is not None else None,
                    gt_label_name=gt_name,
                    pred_label_name=pred_name,
                    gt_bbox=batch.get("bboxes", [None]*len(preds))[i].to("cpu") if batch.get("bboxes") is not None else None,
                    pred_bbox=out.bbox_logits[i].to("cpu"),
                    processor=processor,
                    save_path=os.path.join(save_dir, f"b{b_idx}_s{i}.png") if save_dir else None,
                    title_prefix=f"Batch {b_idx} Sample {i}",
                )
        # calculate classification accuracy
        acc = correct / len(test_set)
        print(f"Test Classification Accuracy: {acc*100:.2f}% ({correct}/{len(test_set)})")

def main():
    model, config, processor = load_models()
    print_model_summary(model)
    test_fish_segmentation_model(model, processor, batch_size=8, max_vis=2, save_dir='./img')

if __name__ == "__main__":
    main()

