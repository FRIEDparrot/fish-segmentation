from transformers import AutoProcessor, DetrForSegmentation, AutoConfig
from Fish_Pretrain.utils import visualize_sample_comparison
from Fish_Pretrain import get_fish_classes
import os
from torchinfo import summary  # for model summary
from Fish_Finetune.fine_tune import load_dataset, FishCollator
import torch
from torch.utils.data import DataLoader
from transformers import BatchFeature
# Load model directly
from transformers import AutoModel

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#region Model Loading and Summary
def load_model(model_path: str):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = DetrForSegmentation.from_pretrained(model_path, config=config, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    return model, processor, config

def summary_model(model):
    batch_size = 1
    channels = 3
    height = 512  # typical DETR input size, can be adjusted
    width = 512
    pixel_values = torch.randn(batch_size, channels, height, width)
    pixel_mask = torch.ones(batch_size, height, width)
    summary(model, input_data={"pixel_values": pixel_values, "pixel_mask": pixel_mask})
#endregion

def move_batch_to_device(batch, device):
    """
    Recursively moves a batch (Tensor, dict, list, BatchFeature) to a device.
    Modifies the input in place when possible.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    elif isinstance(batch, list):
        return [move_batch_to_device(x, device) for x in batch]

    elif isinstance(batch, (dict, BatchFeature)):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    else:
        # non-tensor values (ints, strings, None, etc.)
        return batch

def evaluate_model(
        model, processor, batch_size: int = 1, max_vis: int = 2, save_dir: str = None
    ):
    _, test_set = load_dataset(train_size=0.95)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,
                             collate_fn=FishCollator(processor), pin_memory=True if torch.cuda.is_available() else False)
    classes = get_fish_classes()
    model.eval()
    model.to(device)

    with torch.no_grad():
        show_num = 0
        correct = 0
        for b_idx, batch in enumerate(test_loader):
            # since there is nested structure in inputs, only pin_memory for tensors is not enough
            inputs = move_batch_to_device(batch, device)
            outputs = model(**inputs)

            images = inputs['pixel_values']
            labels = [label for label in inputs['labels']]

            # ========   predict result (written by GPT) ===============
            class_pred = outputs["logits"].argmax(-1)  # (B, num_queries)
            pred_masks = outputs["pred_masks"].sigmoid()  # (B, num_queries, H, W)
            pred_bboxes = outputs["pred_boxes"]  # (B, num_queries, 4)
            if show_num >= max_vis:
                print("calculating accuracy ...", "batch: ", b_idx, "/", len(test_loader))

            batch_size = images.shape[0]   # the bbox is predicted from mask
            # calculate the classification accuracy -> gt : ground truth
            for i in range(batch_size):
                img = images[i]  # shape [3, H, W]
                label = labels[i] # predicted labels
                # --- Handle predicted class, masks and bboxes (they should all be 1)---
                class_ids = label['class_labels']  # input class ids
                masks = label['masks']  # [B, 800, 1060]
                bboxes = label['boxes']  # NOTE : the bbox is normalized
                if len(class_ids) > 1:
                    print(f"Warning: Multiple classes detected in sample {i}, using the first.")
                if len(masks) > 1:
                    print(f"Warning: Multiple masks detected in sample {i}, using the first.")
                if len(bboxes) > 1:
                    print(f"Warning: Multiple boxes detected in sample {i}, using the first.")

                gt_label = class_ids[0] if len(class_ids) > 0 else None
                gt_mask = masks[0] if len(masks) > 0 else None
                gt_box = bboxes[0] if len(bboxes) > 0 else None

                # --- Prediction Extraction ---
                valid_idx = class_pred[i] != model.config.num_labels
                filtered_classes = class_pred[i][valid_idx]
                filtered_masks = (pred_masks[i][valid_idx] > 0.5).long()   # for sigmoid function, we need to compare to 0.5
                filtered_boxes = pred_bboxes[i][valid_idx]

                # collate predicted results
                pred_label = filtered_classes[0] if filtered_classes.nelement() > 0 else None
                if pred_label == gt_label:
                    correct += 1

                if show_num >= max_vis:
                    continue
                else:
                    show_num += 1
                pred_mask = filtered_masks[0] if filtered_masks.nelement() > 0 else None
                pred_box = filtered_boxes[0] if filtered_boxes.nelement() > 0 else None
                # get label names
                gt_label_name = classes[gt_label] if gt_label is not None else "N/A"
                pred_label_name = classes[pred_label] if pred_label else "N/A"
                visualize_sample_comparison(
                    img.cpu(),
                    gt_mask.cpu(), pred_mask.cpu(),
                    gt_label_name, pred_label_name,
                    gt_box.cpu(), pred_box.cpu(),
                    save_path=os.path.join(save_dir, f"b{b_idx}_s{i}.png") if save_dir else None,
                    bbox_mode="center",
                )
        # compute accuracy
        total = len(test_set)
        accuracy = correct / total # total number of samples in the test set
        print(f"Classification Accuracy on Test Set: {accuracy*100:.2f}% ({correct}/{total})")

def main():
    model_path: str = "./models/final_model"
    model, processor, config = load_model(model_path)
    # summary_model(model)
    evaluate_model(model, processor, max_vis=5, save_dir="./img")

if __name__ == "__main__":
    main()
