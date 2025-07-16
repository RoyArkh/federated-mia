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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def convert_huggingface_data_to_list_dic(dataset):
    print(f"Converting HuggingFace dataset to list...")
    return [dataset[i] for i in range(len(dataset))]

def load_model(name, ref=False):
    print(f"Loading {'reference' if ref else 'target'} model: {name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")

def load_ref_model(attacked_model_name):
    try:
        if "mobilellm" in attacked_model_name.lower():
            return load_model('facebook/MobileLLM-1.5B', ref=True)
        elif "pythia" in attacked_model_name.lower():
            return load_model('EleutherAI/pythia-2.8b', ref=True)
        elif "gpt-neo" in attacked_model_name.lower():
            return load_model('EleutherAI/gpt-neo-2.7B', ref=True)
        else:
            raise ValueError("No reference model matched.")
    except Exception as e:
        raise RuntimeError(f"Failed to load reference model: {e}")

def load_benchmark_dataset(dataset_name, dataset_split):
    try:
        print(f"Loading dataset: {dataset_name}, split: {dataset_split}")
        if dataset_name == "ag_news":
            return load_dataset('mia-llm/ag_news_benchmark_roya', split=dataset_split)
        elif dataset_name == "xsum":
            return load_dataset('mia-llm/xsum_benchmark_roya', split=dataset_split)
        elif dataset_name == "wikitext":
            return load_dataset('mia-llm/wikitext2raw_benchmark_roya', split=dataset_split)
        else:
            print("Custom dataset detected — loading from Hugging Face Hub...")
            return load_dataset(dataset_name, split=dataset_split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

def inference(text, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return -loss.item()

def run_attacks(scores, data, model, tokenizer, ref_model, ref_tokenizer):
    print("Running attacks...")
    for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')):
        text = d['text']

        try:
            input_ids = tokenizer.encode(text)
            if len(input_ids) < 3:
                print(f"[SKIP] Sample {i} too short: '{text[:50]}'")
                continue

            input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_ids_tensor, labels=input_ids_tensor)
            loss, logits = outputs[:2]

            if torch.isnan(loss):
                print(f"[NaN LOSS] at sample {i}: '{text[:50]}'")
                continue

            ll = -loss.item()
            ll_ref = inference(text, ref_model, ref_tokenizer)
            ll_lower = inference(text.lower(), model, tokenizer)

            scores['ref'].append(ll - ll_ref)
            scores['lowercase'].append(ll_lower / ll if ll != 0 else np.nan)
            scores['loss'].append(ll)
            scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

            input_ids = input_ids_tensor[0][1:].unsqueeze(-1)
            probs = F.softmax(logits[0, :-1], dim=-1)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)

            if len(token_log_probs) == 0:
                print(f"[SKIP] Empty token_log_probs at sample {i}")
                continue

            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

            for ratio in [0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k = int(len(token_log_probs) * ratio)
                if k > 0:
                    topk = np.sort(token_log_probs.cpu())[:k]
                    scores[f'mink_{ratio}'].append(np.mean(topk).item())
                else:
                    scores[f'mink_{ratio}'].append(np.nan)

            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k = int(len(mink_plus) * ratio)
                if k > 0:
                    topk = np.sort(mink_plus.cpu())[:k]
                    scores[f'mink++_{ratio}'].append(np.mean(topk).item())
                else:
                    scores[f'mink++_{ratio}'].append(np.nan)

            for w in [2, 3, 4, 5, 6, 7]:
                spans = [token_log_probs[i:i+w].mean().item() for i in range(len(token_log_probs) - w + 1)]
                sorted_spans = np.sort(spans)
                for ratio in [0.05, 0.1, 0.2, 0.3]:
                    k = int(len(sorted_spans) * ratio)
                    if k > 0:
                        scores[f'wink_{w}_ratio_{ratio}'].append(np.mean(sorted_spans[:k]))
                    else:
                        scores[f'wink_{w}_ratio_{ratio}'].append(np.nan)

        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            continue

def get_metrics(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)

    clean = [(s, l) for s, l in zip(scores, labels) if not np.isnan(s)]
    if len(clean) == 0:
        print("[WARNING] No valid (non-NaN) scores found.")
        return np.nan, np.nan, np.nan

    scores_clean, labels_clean = zip(*clean)
    scores_clean = np.array(scores_clean)
    labels_clean = np.array(labels_clean)

    if len(np.unique(labels_clean)) < 2:
        print("[WARNING] Not enough class variety after cleaning.")
        return np.nan, np.nan, np.nan

    try:
        fpr, tpr, _ = roc_curve(labels_clean, scores_clean)
        auroc = auc(fpr, tpr)
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
        tpr05 = tpr[np.where(fpr <= 0.05)[0][-1]]
        return auroc, fpr95, tpr05
    except Exception as e:
        print(f"[WARNING] ROC/AUC calculation failed: {e}")
        return np.nan, np.nan, np.nan


def make_file_and_save(scores, data, save_folder, attacked_model_name):
    print("Saving results...")
    labels = [d['label'] for d in data]
    results = defaultdict(list)

    for method, score in scores.items():
        auroc, fpr95, tpr05 = get_metrics(score, labels)
        results['method'].append(method)
        results['auroc'].append(f"{auroc:.1%}" if not np.isnan(auroc) else "NaN")
        results['fpr95'].append(f"{fpr95:.1%}" if not np.isnan(fpr95) else "NaN")
        results['tpr05'].append(f"{tpr05:.1%}" if not np.isnan(tpr05) else "NaN")

    df_summary = pd.DataFrame(results)
    df_full = pd.DataFrame.from_dict(scores)

    print(df_summary)

    os.makedirs(save_folder, exist_ok=True)
    model_id = attacked_model_name.split("/")[-1]

    summary_path = os.path.join(save_folder, f"{model_id}.csv")
    full_path = os.path.join(save_folder, f"{model_id}_full_scores.csv")

    df_summary.to_csv(summary_path, index=False)
    df_full.to_csv(full_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset_split', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    ref_model, ref_tokenizer = load_ref_model(args.model)

    dataset = load_benchmark_dataset(args.root_dataset, args.dataset_split)
    data = convert_huggingface_data_to_list_dic(dataset)

    scores = defaultdict(list)
    run_attacks(scores, data, model, tokenizer, ref_model, ref_tokenizer)
    make_file_and_save(scores, data, args.save_folder, args.model)

if __name__ == "__main__":
    main()
