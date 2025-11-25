from huggingface_hub import HfApi

# import token from .env
from dotenv import load_dotenv
import os

def deploy():
    print(">>> Phase 3: Deploying Model to HuggingFace Hub")
    api = HfApi()
    repo_id = "sirasira/flood-lstm-v1"
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    files = ["pytorch_model.bin", "config.json", "scaler.pkl", "thresholds.json"]

    for f in files:
        if os.path.exists(f):
            print(f"   Uploading {f}...")
            api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo_id, token=token)
        else:
            print(f"\u274C Missing {f}. Did you run Phases 1 & 2?")

    print("\u2705 Phase 3 Complete. Production Ready.")

if __name__ == "__main__":
    deploy()
