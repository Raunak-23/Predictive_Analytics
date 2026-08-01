from pathlib import Path
import shutil

import kagglehub
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(handle: str, target_filename: str, destination_filename: str | None = None) -> Path | None:
    destination_filename = destination_filename or target_filename
    print(f"Downloading Kaggle dataset: {handle}")

    try:
        downloaded_path = kagglehub.dataset_download(handle, path=str(DATA_DIR), force_download=True)
    except Exception as exc:
        print(f"Could not download Kaggle dataset {handle}: {exc}")
        return None

    source_dir = Path(downloaded_path)

    if not source_dir.exists():
        print(f"Downloaded dataset path does not exist: {source_dir}")
        return None

    for item in source_dir.iterdir():
        destination = DATA_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    matches = list(source_dir.rglob(target_filename))
    if matches:
        source_file = matches[0]
        final_path = DATA_DIR / destination_filename
        shutil.copy2(source_file, final_path)
        print(f"Saved ready-to-import file: {final_path}")
        return final_path

    print(f"Downloaded dataset contents into: {DATA_DIR}")
    return DATA_DIR


def download_huggingface_dataset(dataset_name: str, config_name: str | None = None, output_folder: str | None = None) -> Path:
    print(f"Downloading Hugging Face dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)

    folder_name = output_folder or dataset_name.replace("/", "_")
    target_dir = DATA_DIR / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target_dir))
    print(f"Saved Hugging Face dataset to: {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_kaggle_dataset(
        "yasirali646/email-intent-classification",
        "email_intent.csv",
        "business_email_intent.csv",
    )

    download_kaggle_dataset(
        "marcelwiechmann/enron-spam-data",
        "enron_spam_data.csv",
    )

    if not (DATA_DIR / "enron_spam_data.csv").exists():
        download_huggingface_dataset("VoltageVagabond/spam-email-dataset", None, "spam_email_dataset")

    download_huggingface_dataset("talby/spamassassin", "unprocessed", "spamassassin")