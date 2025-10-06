from transformers import AutoProcessor, DetrForSegmentation, AutoConfig
from Fish_Pretrain.utils import visualize_sample_comparison
from Fish_Pretrain import get_fish_classes
import os
from torchinfo import summary  # for model summary
from Fish_Finetune.fine_tune import load_dataset, FishCollator
import torch
from torch.utils.data import DataLoader
from transformers import BatchFeature

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
    import torch
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
        model: DetrForSegmentation, processor, batch_size: int = 1, max_vis: int = 2, save_dir: str = None
    ):
    _, test_set = load_dataset(train_size=0.8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,
                             collate_fn=FishCollator(processor), pin_memory=True if torch.cuda.is_available() else False)
    classes = get_fish_classes()
    model.eval()
    model.to(device)
    with torch.no_grad():
        show_num = 0
        for batch in test_loader:
            inputs = move_batch_to_device(batch, device)
            # since there is nested structure in inputs, only pin_memory for tensors is not enough
            outputs = model(**inputs)

            # get original data for visualization
            images = inputs['pixel_values']
            labels = [label for label in inputs['labels']]

            class_ids = [label['class_labels'] for label in labels]
            masks = [label['masks'] for label in labels]
            bboxes = [label['boxes'] for label in labels]

            # predicted result :
            class_pred = outputs["logits"].argmax(-1)  # (B, num_queries)


            if show_num >= max_vis:
                break
            for i in range(len(batch)):
                show_num += 1
                if show_num >= max_vis:
                    break
                img = images[i]
                gt_mask = masks[i][0] if masks[i].nelement() > 0 else None
                gt_label = batch[i]['class_id']

            break  # Remove this break to evaluate the entire test set

def main():
    model_path: str = "./models/final_model"
    model, processor, config = load_model(model_path)
    # summary_model(model)
    evaluate_model(model, processor)

if __name__ == "__main__":
    main()