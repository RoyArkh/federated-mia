import torch
from collections import OrderedDict

def save_model_weights(model, path: str):
    torch.save(model.state_dict(), path)

def load_model_weights(model, path: str):
    state_dict = torch.load(path, map_location=model.device)
    model.load_state_dict(state_dict, strict=False)
