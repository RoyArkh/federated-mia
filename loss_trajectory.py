import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
from huggingface_hub import login
import matplotlib.ticker as ticker
import os
import re

login(token="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")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_rounds = 21
dataset_id = "mia-llm/wikitext2raw_benchmark_roya"
dataset = load_dataset(dataset_id, split="dataset_32")

members = dataset.select([24, 25, 26])
non_members = dataset.select([374, 375, 376])

samples = [
    {"text": ex["text"], "label": "member", "id": f"member_{i}", "losses": []}
    for i, ex in enumerate(members)
] + [
    {"text": ex["text"], "label": "non-member", "id": f"nonmember_{i}", "losses": []}
    for i, ex in enumerate(non_members)
]

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

for round_id in range(1, num_rounds):
    model_id = f"YusufBayram-Personal/EleutherAI-pythia-70m_wikitext_round_{round_id}"
    model = AutoModelForCausalLM.from_pretrained(model_id, token="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").to(device)
    model.eval()

    for ex in samples:
        inputs = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        ex["losses"].append(loss)
        print(f"{ex['id']} - Round {round_id} loss: {loss:.4f}")


plt.figure(figsize=(12, 6))

samples.sort(key=lambda ex: sum(ex["losses"]) / len(ex["losses"]))

for ex in samples:
    color = "C0" if ex["label"] == "member" else "C1"
    plt.plot(
        range(1, len(ex["losses"]) + 1),
        ex["losses"],
        label=f"{ex['id']} ({ex['label']})",
        marker="o",
        color=color
    )

plt.title("Loss over Rounds for Selected Member and Non-Member Samples")
plt.xlabel("Federated Round")
plt.ylabel("Cross-Entropy Loss")

all_losses = [loss for ex in samples for loss in ex["losses"]]
min_loss = min(all_losses)
plt.ylim(bottom=min_loss)


plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.legend()
plt.grid(True)
plt.tight_layout()


os.makedirs("plots", exist_ok=True)

existing_plots = [
    f for f in os.listdir("plots")
    if re.match(r"loss_trends_mia_\d+\.png", f)
]
plot_numbers = [
    int(re.search(r"(\d+)", f).group(1)) for f in existing_plots
]
next_plot_number = max(plot_numbers) + 1 if plot_numbers else 1

filename = f"loss_trends_mia_{next_plot_number}.png"
save_path = os.path.join("plots", filename)


plt.savefig(save_path)
plt.show()

