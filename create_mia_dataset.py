import argparse
import pickle
from datasets import Dataset, load_dataset

def load_client_data(client_id, data_dir="data", set_type="train", num_samples=350):
    path = f"{data_dir}/client_{client_id}_{set_type}.pkl"
    with open(path, "rb") as f:
        texts = pickle.load(f)    
        return texts[:num_samples]

def load_non_member_data(dataset_name):
    if dataset_name == "xsum":
        name = "mia-llm/xsum_benchmark_roya"
    elif dataset_name == "wikitext":
        name = "mia-llm/wikitext2raw_benchmark_roya"
    elif dataset_name == "ag_news":
        name = "mia-llm/ag_news_benchmark_roya"

    dataset = load_dataset(name, split="dataset_32")
    non_member_dataset = dataset.filter(lambda example: example["label"] == 0)
    print(non_member_dataset[0])
    texts = non_member_dataset["text"]
    return texts[:350]

def build_mia_dataset(member_texts, non_member_texts):
    all_texts = member_texts + non_member_texts
    all_labels = [1] * len(member_texts) + [0] * len(non_member_texts)

    return Dataset.from_dict({
        "text": all_texts,
        "label": all_labels
    })    

def main():
    parser = argparse.ArgumentParser(description="Create and push full MIA dataset")
    parser.add_argument("--member_client_id", type=int, required=True)
    parser.add_argument("--root_dataset", type=str, required=True, choices=["xsum", "wikitext", "ag_news"])
    

    args = parser.parse_args()

    member_texts = load_client_data(args.member_client_id)
    non_member_texts = load_non_member_data(args.root_dataset)

    mia_dataset = build_mia_dataset(member_texts, non_member_texts)

    repo_name = f"mia_{args.root_dataset}_client{args.member_client_id}"
    full_repo_name = f"YusufBayram-Personal/{repo_name}"

    

    print(f"Pushing dataset to Hugging Face Hub..")
    mia_dataset.push_to_hub(full_repo_name)
    print("Upload completed")

if __name__ == "__main__":
    main()
