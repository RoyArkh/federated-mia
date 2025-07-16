import torch

def fedprox(weight_paths: list, global_weights_path: str, mu: float = 0.001):

    global_state_dict = torch.load(global_weights_path, map_location="cpu")

    avg_state_dict = None
    for path in weight_paths:
        state_dict = torch.load(path, map_location="cpu")
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for key in avg_state_dict:
                avg_state_dict[key] += state_dict[key].float()

    for key in avg_state_dict:
        avg_state_dict[key] /= len(weight_paths)

        prox_penalty = mu * (global_state_dict[key].float() - avg_state_dict[key])
        avg_state_dict[key] += prox_penalty

    return avg_state_dict 