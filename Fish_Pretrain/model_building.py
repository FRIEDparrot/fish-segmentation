from typing import Union
from transformers import AutoModelForImageSegmentation, AutoImageProcessor
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
import warnings
import torch
from transformers import Trainer, TrainingArguments
from Fish_Pretrain.dataset_loading import load_fish_datasets_all, get_fish_classes, FishSegmentDataCollator, resize_mask
from Fish_Pretrain.decoder_head import FishSegmentBBoxHead, FishSegmentClassifier, FishSegmentationHead
from torch.nn import functional as F
from torch import nn
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LOSS_WEIGHTS = dict(cls=2.0, seg=1.0, box=1.0)  # weights for classification, segmentation, bbox losses

# Close all warnings
warnings.filterwarnings("ignore")


# region Model Definition

@dataclass
class FishSegmentationModelOutput(ModelOutput):
    """
    All members must be Tensor type
    """
    loss: Optional[torch.Tensor] = None
    class_logits: Optional[torch.Tensor] = None
    seg_logits: Optional[torch.Tensor] = None
    bbox_logits: Optional[torch.Tensor] = None


class FishSegmentModelConfig(PretrainedConfig):
    model_type = "fish_segmentation_model"

    def __init__(self, model_name: str = "fish_segmentation_model", hidden_ch=256, num_labels: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_ch = hidden_ch
        if num_labels is not None:
            self.num_labels = num_labels


class FishSegmentationModel(PreTrainedModel):
    config_class = FishSegmentModelConfig

    def __init__(self,
                 model_name: str = "fish_segmentation_model",
                 config: Optional[FishSegmentModelConfig] = None):
        if config is None:
            config = FishSegmentModelConfig(model_name=model_name)
        super().__init__(config)
        # get base model from huggingface
        model = AutoModelForImageSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic", device_map="auto", dtype="auto"
        )
        classes = get_fish_classes()
        num_classes = len(classes)
        model.config.num_labels = num_classes  # Update this based on your dataset

        # extract resnet50 pretrained backbone, also re-use the bbox head
        backbone = model.detr.model.backbone  # (B, 256, W, H)

        self.model_name = model_name
        self.backbone = backbone

        self.conv = nn.Conv2d(2048, config.hidden_ch, kernel_size=1)  # reduce channels from 2048 to hidden_ch
        # much smaller resolution is fine for classification & bbox regression, but quite coarse for pixel segmentation.
        self.classifier = FishSegmentClassifier(in_ch=config.hidden_ch,
                                                num_classes=num_classes)  # (hidden_ch -> num_classes) classifier
        self.bbox_head = FishSegmentBBoxHead(config.hidden_ch)  # (hidden_ch -> 4) bbox head
        self.segmentation_head = FishSegmentationHead(in_ch=config.hidden_ch, num_classes=num_classes)

    def forward(self,
                pixel_values: torch.Tensor,
                pixel_mask: torch.Tensor,
                class_ids: Optional[torch.Tensor] = None,
                segmentation_mask: Optional[torch.Tensor] = None,
                bboxes: Optional[torch.Tensor] = None,
                **kwargs,  # meta info (not used here)
                ) -> FishSegmentationModelOutput:
        """
        Not use "batch" for forward, this is firstly for explicitness, secondly for compatibility with Trainer
        """
        # all hidden states and positional encodings are computed in the backbone
        features_list, masks_list = self.backbone(pixel_values, pixel_mask)
        # Note W, H is 25, 34, (original  is 800, 1060)
        feat, _ = features_list[-1]  # (B, 2048, W, H), use the last feature map
        feat = self.conv(feat)  # (B, hidden_ch, W, H)
        B, C, W_feat, H_feat = feat.shape

        # calculate output of each head
        class_logits = self.classifier(feat)  # (B, num_classes+1)
        bbox_logits = self.bbox_head(feat)  # (B, 4)
        seg_logits = self.segmentation_head(feat, seg_output_size=(W_feat, H_feat))  # (B, num_classes+1, W, H)

        out = {
            "class_logits": class_logits,
            "seg_logits": seg_logits,
            "bbox_logits": bbox_logits,
        }
        if class_ids is not None and segmentation_mask is not None and bboxes is not None:
            loss = self.segmentation_loss_func(
                class_ids=class_ids,
                segmentation_mask=segmentation_mask,
                bboxes=bboxes,
                class_logits=class_logits,
                seg_logits=seg_logits,
                bbox_logits=bbox_logits,
                loss_weights=DEFAULT_LOSS_WEIGHTS,
                return_outputs=False,
            )
            # not only 3 construct and use out.loss = loss, this will not be recorded
            out["loss"] = loss
        return FishSegmentationModelOutput(**out)

    @staticmethod
    def segmentation_loss_func(
            class_ids, segmentation_mask, bboxes,
            class_logits, seg_logits, bbox_logits,
            loss_weights=None,
            return_outputs: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, FishSegmentationModelOutput]]:
        """
        Custom Multi-task loss function combining classification, segmentation, and bounding box losses.

        use outputs = model(**inputs) if modified to compute_loss signature
        """
        if loss_weights is None:
            loss_weights = DEFAULT_LOSS_WEIGHTS

        # calculate individual losses
        cls_loss = F.cross_entropy(class_logits, class_ids)  # Classification CE
        box_loss = F.l1_loss(bbox_logits, bboxes)  # Bounding box L1

        # for segmentation, use different resolution
        W_feat, H_feat = seg_logits.shape[2], seg_logits.shape[3]  # (B, num_classes, W_feat, H_feat)
        seg_targets = torch.stack(
            [resize_mask(mask, (W_feat, H_feat)) for mask in segmentation_mask])  # resize to match seg_logits
        seg_loss = F.cross_entropy(seg_logits, seg_targets, ignore_index=255)  # no ignore now; keep for extensibility
        tot_loss = (loss_weights["cls"] * cls_loss +
                    loss_weights["seg"] * seg_loss +
                    loss_weights["box"] * box_loss)  # weighted sum

        outputs = FishSegmentationModelOutput(
            loss=tot_loss,
            class_logits=class_logits,
            seg_logits=seg_logits,
            bbox_logits=bbox_logits,
        )
        return (tot_loss, outputs) if return_outputs else tot_loss


# Register the model and config to Hugging Face Auto classes
try:
    AutoConfig.register("fish_segmentation_model", FishSegmentModelConfig)
except ValueError:
    pass
try:
    AutoModel.register(FishSegmentModelConfig, FishSegmentationModel)
except ValueError:
    pass

# endregion


# region Trainer Definition (TODO)
# ---------------------------
# Custom Trainer
# ---------------------------
class MultiTaskTrainer(Trainer):
    def __init__(self, loss_weights=None, *args, **kwargs):
        super(MultiTaskTrainer, self).__init__(*args, **kwargs)
        if loss_weights is None:
            loss_weights = DEFAULT_LOSS_WEIGHTS
        self.loss_weights = loss_weights


# endregion

def train_model():
    train_size, test_size = 0.75, 0.2
    config = FishSegmentModelConfig(model_name="FishSegmentationModel", hidden_ch=256)
    training_args = TrainingArguments(
        output_dir='./models',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        label_names=["class_ids", "segmentation_mask", "bboxes"],
        do_train=True,  # train_set
        do_eval=True,  # val_set
        do_predict=True,  # test_set
        eval_strategy="steps",
        weight_decay=1e-2,
        num_train_epochs=20,  # 20 epoch for training
        learning_rate=2e-4,  # specify learning rate
    )

    model = FishSegmentationModel(model_name="fish_segmentation_model", config=config).to(DEVICE)
    train_set, eval_set, test_set = load_fish_datasets_all(train_size, test_size)

    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic", use_fast=True)
    # here we specify datasets and data collator
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=FishSegmentDataCollator(processor=processor),
        compute_metrics=None,  # can define custom metrics if needed
    )
    trainer.train()
    trainer.save_model('./models/final_model')
    processor.save_pretrained('./models/final_model')
    config.save_pretrained('./models/final_model')

    repo_id = "FriedParrot/fish-segmentation-model"

    # push model to hub
    config.push_to_hub(
        repo_id=repo_id,
        private=False,
        commit_message="Add config for FishSegmentationModel"
    )

    processor.push_to_hub(
        repo_id=repo_id,
        private=False,
        commit_message="Add image processor"
    )

    model.push_to_hub(
        repo_id=repo_id,
        private=False,
        commit_message="Add trained FishSegmentationModel"
    )
    print("Model training and saving completed.")

def main():
    train_model()

if __name__ == '__main__':
    main()
