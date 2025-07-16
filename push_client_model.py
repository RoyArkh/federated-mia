import argparse
import os
from model_utils import load_model_and_tokenizer
from weights_utils import load_model_weights

def push_client_model(model_id, client_id, round_id, repo_name, weights_dir="weights"):
    model, tokenizer = load_model_and_tokenizer(model_id)

    weight_path = os.path.join(weights_dir, f"client_{client_id}_round_{round_id}.pt")
    load_model_weights(model, weight_path)

    print(f"pushing model to huggingface Hub: {repo_name}")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print("Model and tokenizer pushed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Push a trained client model to Hugging Face Hub")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--round_id", type=int, required=True)

    args = parser.parse_args()

    repo_name = f"RoyArkh/Test1-{args.model_id.replace('/', '-')}_client{args.client_id}_round{args.round_id}"

    push_client_model(
        model_id=args.model_id,
        client_id=args.client_id,
        round_id=args.round_id,
        repo_name=repo_name
    )

if __name__ == "__main__":
    main()
