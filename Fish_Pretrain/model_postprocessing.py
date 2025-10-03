from torchsummary import summary
from transformers import AutoConfig, AutoModel
import torch


def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def print_model_summary(model, input_size):
    summary(model, input_size)
