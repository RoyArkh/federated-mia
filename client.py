import pickle
import os 
from datasets import Dataset
from model_utils import load_model_and_tokenizer
from trainer_utils import create_trainer
from weights_utils import save_model_weights, load_model_weights

def run_client(client_id: int, round_id: int, model_id: str, dataset_id: str, strategy: str):

    print(f"Client [{client_id}] Round [{round_id}] started")

    train_path = os.path.join("data", f"client_{client_id}_train.pkl")
    eval_path = os.path.join("data", f"client_{client_id}_eval.pkl")

    with open(train_path, "rb") as f:
        train_texts = pickle.load(f)
    with open(eval_path, "rb") as f:
        eval_texts = pickle.load(f)

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    model, tokenizer = load_model_and_tokenizer(model_id)

    global_weights_path = os.path.join("weights", f"global_round_{round_id}.pt")
    load_model_weights(model, global_weights_path)

    output_dir = os.path.join("logs", f"client_{client_id}_round_{round_id}")
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir)
    trainer.train()

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)

    os.makedirs("evaluation_results", exist_ok=True)

    log_path = f"evaluation_results/{strategy}_log.csv"
    with open(log_path, "a") as f:
        f.write(f"{client_id},{round_id},{eval_loss}\n")

    clients_weights_path = os.path.join("weights", f"client_{client_id}_round_{round_id}.pt")
    save_model_weights(model, clients_weights_path)

    print(f"Client {client_id} round {round_id} finished.")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--round_id", type=int, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    args = parser.parse_args()

    run_client(
        client_id=args.client_id,
        round_id=args.round_id,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        strategy=args.strategy
    )