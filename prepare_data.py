from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pickle
import os

def save_client_dataset(dataset_id: str, save_dir: str, num_clients: int):

    if dataset_id == "wikitext":
        dataset = load_dataset("mia-llm/wikitext2-raw-MIA")
    elif dataset_id == "ag_news":
        dataset = load_dataset("mia-llm/AGnews-raw-MIA")
    elif dataset_id == "xsum":
        dataset = load_dataset("mia-llm/xsum-raw-MIA")
    else:
        raise ValueError("Unknown dataset id")
    
    dataset = dataset["train"]

    os.makedirs(save_dir, exist_ok=True)

    for client_id in range(num_clients):
        client_data = dataset.shard(num_shards=num_clients, index=client_id)
        texts = [t.strip() for t in client_data['text'] if t.strip()]

        train_data, eval_data = train_test_split(texts, test_size=0.1, random_state=42)

        with open(f"{save_dir}/client_{client_id}_train.pkl", "wb") as f:
            pickle.dump(train_data, f)
        with open(f"{save_dir}/client_{client_id}_eval.pkl", "wb") as f:
            pickle.dump(eval_data, f)

    
    print(f"Client datasets saved to {save_dir}")



    
