import torch

momentum = {}
variance = {}

def fedadam(weight_paths: list,
    global_weights_path: str,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
    ):

    global momentum, variance
    global_state_dict = torch.load(global_weights_path, map_location="cpu")
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

    updated_state_dict =  {}
    for key in avg_state_dict:
        update = avg_state_dict[key] - global_state_dict[key].float()

        if key not in momentum:
            momentum[key] = torch.zeros_like(update)
            variance[key] = torch.zeros_like(update)        
              
        momentum[key] = beta1 * momentum[key] + (1 - beta1) * update
        variance[key] = beta2 * variance[key] + (1 - beta2) * update

        m_hat = momentum[key] / (1 - beta1)
        v_hat = variance[key] / (1- beta2)

        updated_state_dict[key] = global_state_dict[key].float() + lr * m_hat / (torch.sqrt(v_hat) + eps)

    return updated_state_dict


    