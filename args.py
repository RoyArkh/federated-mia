import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="federated Learning")

    parser.add_argument("--model_id", type=str, required=True, choices=[
        "facebook/MobileLLM-125M", 
        "facebook/MobileLLM-350M",
        "facebook/MobileLLM-600M",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/gpt-neo-125m", #
    ])

    parser.add_argument("--dataset_id", type=str, required=True, choices=[
        "ag_news", #
        "wikitext",
        "xsum"
    ])
 
    parser.add_argument("--num_clients", type=int, default=50)

    parser.add_argument("--num_rounds", type=int, default=3)

    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox", "fedadam"])
    
    return parser.parse_args()
