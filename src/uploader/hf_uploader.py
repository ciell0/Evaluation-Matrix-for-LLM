import os
from huggingface_hub import HfApi, HfFolder, upload_folder, upload_file


class HFUploader:
    """
    Utility class untuk upload model, dataset, atau files ke Hugging Face Hub.
    """

    def __init__(self, repo_id: str, token: str = None, private: bool = True):
        self.repo_id = repo_id
        self.token = token or HfFolder.get_token()

        if self.token is None:
            raise ValueError("HuggingFace token tidak ditemukan. "
                             "Set token menggunakan: huggingface-cli login")

        self.api = HfApi()

        # buat repo jika belum ada
        self.api.create_repo(
            name=repo_id,
            token=self.token,
            private=private,
            exist_ok=True
        )

    def upload_model_folder(self, model_dir: str):
        """
        Upload seluruh folder model (model files, tokenizer, config.json, dsb)
        """
        if not os.path.isdir(model_dir):
            raise ValueError(f"{model_dir} tidak ditemukan")

        print(f"Uploading model folder: {model_dir} → {self.repo_id}")
        upload_folder(
            folder_path=model_dir,
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token
        )
        print("Upload model selesai.\n")

    def upload_file(self, filepath: str, path_in_repo: str = None):
        """
        Upload file tunggal, misal hasil evaluasi, logs, csv, jsonl.
        """
        if not os.path.isfile(filepath):
            raise ValueError(f"File {filepath} tidak ditemukan")

        print(f"Uploading file: {filepath} → {self.repo_id}")
        upload_file(
            path_or_fileobj=filepath,
            path_in_repo=path_in_repo or os.path.basename(filepath),
            repo_id=self.repo_id,
            token=self.token,
            repo_type="model"
        )
        print("Upload file selesai.\n")

    def upload_dataset_folder(self, data_dir: str):
        """
        Upload folder dataset (misal .jsonl)
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"{data_dir} tidak ditemukan")

        print(f"Uploading dataset folder: {data_dir} → {self.repo_id}")
        upload_folder(
            folder_path=data_dir,
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token
        )
        print("Upload dataset selesai.\n")



# -------------------------
# Contoh Pemakaian
# -------------------------
if __name__ == "__main__":
    uploader = HFUploader(
        repo_id="username/eval-sft-demo",
        token=os.getenv("HF_TOKEN"),
        private=True
    )

    # upload folder model
    uploader.upload_model_folder("outputs/my-finetuned-model")

    # upload satu file hasil evaluasi
    uploader.upload_file("results/metrics.json")

    # upload dataset folder
    uploader.upload_dataset_folder("data/eval_set")
