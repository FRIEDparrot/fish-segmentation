# import torch
# from Fish_Pretrain import build_dataloaders
# from Fish_Pretrain.model_building import FishSegmentationModel, segmentation_loss
#
#
# def test_compute_loss():
#     """
#     Brief test for compute_loss function using dummy inputs and a minimal FishSegmentationModel.
#     Checks that the function returns a scalar loss and output dict.
#     """
#     # Setup
#     num_classes = 3
#     batch_size = 2
#     H, W = 25, 34
#     device = torch.device('cpu')
#
#     # load model
#     model = FishSegmentationModel(model_name="facebook/detr-resnet-50-panoptic")
#     model.to(device)
#     model.eval()
#
#     # load dataset
#     dataloader, _, _ = build_dataloaders(batch_size=batch_size)
#     inputs = next(iter(dataloader))
#     print("Input keys:", inputs.keys())
#
#     # Test compute_loss
#     loss, outputs = segmentation_loss(model, inputs.copy(), return_outputs=True)
#     print("Loss:", loss)
#     print("Outputs keys:", outputs.keys())
#     assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
#     assert "class_logits" in outputs and "segmentation" in outputs and "bboxes" in outputs, "Missing output keys"
#     print("test_compute_loss passed.")
#
#
# if __name__ == "__main__":
#     test_compute_loss()
