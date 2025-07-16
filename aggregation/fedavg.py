import torch

def fedavg(weight_paths: list):
    avg_state_dict = None

    for idx, path in enumerate(weight_paths):
            state_dict = torch.load(path, map_location="cpu")
            if avg_state_dict is None:
                avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
            else:
                for key in avg_state_dict:
                    avg_state_dict[key] += state_dict[key].float()

    for key in avg_state_dict:
          avg_state_dict[key] /= len(weight_paths)

    return avg_state_dict


                     
