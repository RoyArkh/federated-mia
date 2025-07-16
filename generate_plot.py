import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_strategy_graph(model_id, dataset_id, num_clients, num_rounds, strategy):
    log_path = f"evaluation_results/{strategy}_log.csv"

    if not os.path.exists(log_path):
        print(f"[ERROR] Log file not found: {log_path}")
        return

    df = pd.read_csv(log_path, header=None, names=["client_id", "round_id", "eval_loss"])
    df = df.dropna()
    df["round_id"] = df["round_id"].astype(int)
    df["eval_loss"] = df["eval_loss"].astype(float)

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

    print(f"[INFO] Plot saved to: {save_path}")

# 🔧 Bunları elle gir ya da argparse ile al
plot_strategy_graph(
    model_id="EleutherAI/pythia-410m",
    dataset_id="ag_news",
    num_clients=20,
    num_rounds=3,
    strategy="fedadam"
)
