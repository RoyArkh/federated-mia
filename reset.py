import shutil
import os

def safe_rmdir(path):
    if os.path.exists(path):
        print(f"Removing: {path}")
        shutil.rmtree(path)
    else:
        print(f"Skipping (not found): {path}")

if __name__ == "__main__":
    folders_to_clean = [
        "weights",            # global and client model weights
        "logs",               # trainer logs
        "evaluation_results", # evaluation loss log
        "hf_upload",          # intermediate Hugging Face upload dirs
        "data"                # client dataset shards
    ]

    for folder in folders_to_clean:
        safe_rmdir(folder)

    print("\nEnvironment reset completed.")
