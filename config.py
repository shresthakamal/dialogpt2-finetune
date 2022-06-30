from pathlib import Path


BASE_DIR = ""

DATA_DIR = Path(BASE_DIR, "data", "ijcnlp_dailydialog")

if __name__ == "__main__":
    print(DATA_DIR)