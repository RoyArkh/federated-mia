import os 
import shutil
import torch
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, create_repo, upload_folder
from args import parse_args
from model_utils import load_model_and_tokenizer
from weights_utils import save_model_weights, load_model_weights
from prepare_data import save_client_dataset
from aggregation.fedavg import fedavg
from aggregation.fedadam import fedadam
from aggregation.fedprox import fedprox

def initialize_global_model(model_id: str, weights_dir: str):
    model, tokenizer = load_model_and_tokenizer(model_id)
    path = os.path.join(weights_dir, "global_round_0.pt")
    save_model_weights(model, path)
    print("Server initialized global model.")

    repo_name = f"{model_id.replace('/', '-')}_round_0"
    namespace = "RoyArkh"
    full_repo_id = f"{namespace}/{repo_name}"
    hf_model_dir = os.path.join("hf_upload", repo_name)
    os.makedirs(hf_model_dir, exist_ok=True)

    model.save_pretrained(hf_model_dir)
    tokenizer.save_pretrained(hf_model_dir)

    api = HfApi()
    create_repo(repo_id=full_repo_id, exist_ok=True, private=True)
    upload_folder(repo_id=full_repo_id, folder_path=hf_model_dir, repo_type="model")

    print(f"[UPLOAD] Uploaded untrained global_round_0 model to Hugging Face at {full_repo_id}")

def run_clients_in_round(round_id: int, num_clients: int, model_id: str, dataset_id: str, strategy: str):
    for client_id in range(num_clients):
        print(f"Server launching client {client_id} for round {round_id}...")
        subprocess.run([
            "python", "client.py",
            "--client_id", str(client_id),
            "--round_id", str(round_id),
            "--model_id", model_id,
            "--dataset_id", dataset_id,
            "--strategy", strategy
        ])

def aggregate_client_weights(round_id: int, num_clients: int, weights_dir: str, strategy: str, model_id: str, dataset_id: str):
    paths = [os.path.join(weights_dir, f"client_{i}_round_{round_id}.pt") for i in range(num_clients)]

    if strategy == "fedavg":
        avg_state_dict = fedavg(paths)
    elif strategy == "fedadam":
        global_path = os.path.join(weights_dir, f"global_round_{round_id}.pt")
        avg_state_dict = fedadam(paths, global_path)
    elif strategy == "fedprox":
        global_path = os.path.join(weights_dir, f"global_round_{round_id}.pt")
        avg_state_dict = fedprox(paths, global_path, mu=0.001)
    else:
        raise ValueError("Unsupported strategy")

    save_path = os.path.join(weights_dir, f"global_round_{round_id + 1}.pt")
    torch.save(avg_state_dict, save_path)

    model, tokenizer = load_model_and_tokenizer(model_id)
    load_model_weights(model, save_path)

    repo_name = f"{model_id.replace('/', '-')}_{dataset_id}_round_{round_id + 1}"
    namespace = "RoyArkh"
    full_repo_id = f"{namespace}/{repo_name}"
    hf_model_dir = os.path.join("hf_upload", repo_name)
    os.makedirs(hf_model_dir, exist_ok=True)

    model.save_pretrained(hf_model_dir)
    tokenizer.save_pretrained(hf_model_dir)

    api = HfApi()
    create_repo(repo_id=full_repo_id, exist_ok=True, private=True)
    upload_folder(repo_id=full_repo_id, folder_path=hf_model_dir, repo_type="model")

    print(f"[UPLOAD] Uploaded global model of round {round_id + 1} to Hugging Face at {full_repo_id}")
    # sadece alandan tasarruf etmek için global roundlar dışındaki tüm weightleri siler 
#    for path in paths:
#        if os.path.exists(path) and "client" in os.path.basename(path):
#            os.remove(path)

       
def run_federated_learning(model_id: str, dataset_id: str, num_rounds: int, num_clients: int, strategy):
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    save_client_dataset(dataset_id, "data", num_clients)
    initialize_global_model(model_id, "weights")

    for round_id in range(num_rounds):
        run_clients_in_round(round_id, num_clients, model_id, dataset_id, strategy)
        aggregate_client_weights(round_id, num_clients, "weights", strategy, model_id, dataset_id)

    plot_strategy_graph(model_id, dataset_id, num_clients, num_rounds, strategy)
    print("Server training completed.")

def plot_strategy_graph(model_id, dataset_id, num_clients, num_rounds, strategy):
    log_path = f"evaluation_results/{strategy}_log.csv"
    if not os.path.exists(log_path):
        print(f"log_path can not be found.")
        return None

    df = pd.read_csv(log_path, header=None, names=["client_id", "round_id", "eval_loss"])
    avg_loss = df.groupby("round_id")["eval_loss"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_loss.index, avg_loss.values, marker='o', label=strategy)
    plt.title(f"Eval Loss - {model_id} on {dataset_id} ({num_clients} clients, {num_rounds} rounds)")
    plt.xlabel("Round")
    plt.ylabel("Average Eval Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    filename = f"{model_id}_{dataset_id}_{num_clients}c_{num_rounds}r_{strategy}.png"
    filename = filename.replace("/", "-")
    save_path = os.path.join("plots", filename)
    plt.savefig(save_path)
    print("The plot has been generated.")

if __name__ == "__main__":
    args = parse_args()
    run_federated_learning(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        strategy=args.strategy
    )