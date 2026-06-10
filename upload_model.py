"""
Upload the local model/ directory to HuggingFace Hub.
Usage:  python upload_model.py

Set HF_TOKEN in .env before running.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO_ID   = "velislavp/ml-bert-new-detect-bg"
MODEL_DIR = Path(__file__).parent / "model"


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set in .env — add your HuggingFace write token.")

    print(f"Uploading {MODEL_DIR} -> {REPO_ID} ...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
    )
    print("Done. Model is live at https://huggingface.co/" + REPO_ID)


if __name__ == "__main__":
    main()
