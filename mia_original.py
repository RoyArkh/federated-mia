import os, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


device = torch.device("cuda")
print(f"Device: {device}")


# Helper function to convert a HuggingFace dataset into a list of dictionaries
def convert_huggingface_data_to_list_dic(dataset):
    print(f"converting hugginface data to list for dataset {dataset}")

    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]  # Extracting each sample from the dataset
        all_data.append(ex)  # Appending each sample to the list
    return all_data


# Function to load the model (and tokenizer) based on arguments
def load_model(name, ref=False):
    if ref:
        print(f"loading {name} for ref")
    else:
        print(f"loading {name}")

    # Check for MobileLLM-specific handling
    if "MobileLLM" in name:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True#, **int8_kwargs, **half_kwargs
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=True, use_fast=False
            )
            tokenizer.add_special_tokens({
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
            })
        except Exception as e:
            raise RuntimeError(f"Failed to load MobileLLM model or tokenizer: {e}")
    else:
        # Standard model loading for other models
        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, return_dict=True, device_map='auto'#, **int8_kwargs, **half_kwargs
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    # Ensure model is in evaluation mode
    model.eval()
    
    return model, tokenizer


# Load the reference model and tokenizer for reference attack
def load_ref_model(attacked_model_name):
    try:
        if "mobilellm" in attacked_model_name.lower():
            ref_model, ref_tokenizer = load_model('facebook/MobileLLM-1.5B', ref=True)
        elif "pythia" in attacked_model_name.lower():
            ref_model, ref_tokenizer = load_model('EleutherAI/pythia-2.8b', ref=True)
        elif "gpt-neo" in attacked_model_name.lower():
            ref_model, ref_tokenizer = load_model('EleutherAI/gpt-neo-2.7B', ref=True)

        return ref_model, ref_tokenizer
    except Exception as e:
        print("Couldn't load reference Model")
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")
        

def load_benchmark_dataset(dataset_name, dataset_split):
    #hosein has prepared the latest versions, so I am pulling those in here
    try:
        print(f"Loading benchmark dataset according to args.root_dataset: {dataset_name}")
        if dataset_name == "wikitext":
            dataset = load_dataset('mia-llm/wikitext2raw_benchmark_roya',split=dataset_split)
            print('mia-llm/wikitext2raw_benchmark_roya')
        elif dataset_name == "ag_news":
            dataset = load_dataset('mia-llm/ag_news_benchmark_roya',split=dataset_split)
        elif dataset_name == "xsum":
            dataset = load_dataset('mia-llm/xsum_benchmark_roya', split=dataset_split)
        else:
            raise ValueError("wrong name!")
            #or RoyArkh/new_wikitext2_benchmark - crates an error
            #mia-llm/wikitext2-MIA-Benchmark
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load benchmark dataset: {e}")



# Inference function: computes log-likelihood for a given text
def inference(text, model, tokenizer):
    # roya note : where is this tokenizer coming from
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Encode text into token ids
    input_ids = input_ids.to(device)  # Move to GPU
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(input_ids, labels=input_ids)  # Perform forward pass
    loss, logits = outputs[:2]  # Extract loss and logits
    ll = -loss.item()  # Log-likelihood is the negative of the loss
    return ll



def run_attacks(scores, data, model, tokenizer, ref_model, ref_tokenizer):
    #scores = defaultdict(list)
    print("Running attacks")

    for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')):
        text = d['text']
        #print(text)
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        ll = -loss.item() # log-likelihood
        ll_ref = inference(text, ref_model, ref_tokenizer)
        ll_lowercase = inference(text.lower(), model, tokenizer)

        #ref
        scores['ref'].append(ll - ll_ref)
        #lowercase
        scores['lowercase'].append(ll_lowercase / ll)
        #loss
        scores['loss'].append(ll)
        #zlib
        scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

        #for mink and mink++
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        #mink
        for ratio in [0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            scores[f'mink_{ratio}'].append(np.mean(topk).item())

        #mink++
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            scores[f'mink++_{ratio}'].append(np.mean(topk).item())


        #winmink
        for window_size in [2,3,4,5,6,7]:
            # Slide a window over token_log_probs to get "window-level" averages
            sp_scores = []
            for start_idx in range(0, len(token_log_probs) - window_size + 1):
                window_slice = token_log_probs[start_idx : start_idx + window_size]
                window_avg = window_slice.mean()
                sp_scores.append(window_avg.item())
            span_scores = np.array(sp_scores)
            sorted_scores = np.sort(span_scores)
            for ratio in [0.05, 0.1, 0.2, 0.3]:
                k_length = int(len(sorted_scores) * ratio)
                mink_window_scores = sorted_scores[:k_length]
                mink_score = float(np.mean(mink_window_scores))
                scores[f'wink_{window_size}_ratio_{ratio}'].append(mink_score)


# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):
    #print("getting metrics")

    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05



def make_file_and_save(scores, data, save_folder, attacked_model_name):
    print("saving file")

    labels = [d['label'] for d in data] # 1: training, 0: non-training
    results = defaultdict(list)
    for method, score in scores.items():
        auroc, fpr95, tpr05 = get_metrics(score, labels)
        
        results['method'].append(method)
        results['auroc'].append(f"{auroc:.1%}")
        results['fpr95'].append(f"{fpr95:.1%}")
        results['tpr05'].append(f"{tpr05:.1%}")

    output_df = pd.DataFrame.from_dict(scores)
    # display results
    df = pd.DataFrame(results)
    print(df)
    save_root = save_folder
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    #save brief table
    model_id = attacked_model_name.split('/')[-1]
    fname = f"{model_id}.csv"
    if os.path.isfile(os.path.join(save_root, fname)):
        df.to_csv(os.path.join(save_root,fname), index=False, mode='a', header=False)
    else:
        df.to_csv(os.path.join(save_root, fname), index=False)

    #save full scored for drawing
    score_path = f"{model_id}_full_scores.csv"
    if os.path.isfile(os.path.join(save_root, score_path)):
        output_df.to_csv(os.path.join(save_root, score_path), index=False, mode='a', header=False)
    else:
        output_df.to_csv(os.path.join(save_root, score_path), index=False)


def main():
    parser = argparse.ArgumentParser()
    #we will load benchmark based on this
    parser.add_argument('--root_dataset', type=str, choices=['ag_news', 'xsum', 'wikitext'])
    #fine-tuned models for attacking
    parser.add_argument('--model', type=str)
    #split used for the dataset
    parser.add_argument('--dataset_split', type=str)
    #where the resulta are gonna get saved
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    ref_model, ref_tokenizer = load_ref_model(args.model)

    dataset = load_benchmark_dataset(args.root_dataset, args.dataset_split)
    data = convert_huggingface_data_to_list_dic(dataset)
    scores = defaultdict(list)
    run_attacks(scores, data, model, tokenizer, ref_model, ref_tokenizer)
    #run_attacks(scores, data, model, tokenizer, "ref_model", "ref_tokenizer")
    make_file_and_save(scores, data, args.save_folder, args.model)


if __name__ == '__main__': 
    main()
