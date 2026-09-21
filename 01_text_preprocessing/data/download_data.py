"""Download the Kaggle IMDB review dataset without committing the large CSV."""
from pathlib import Path
import shutil
import subprocess
import sys

DATASET_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
DATA_DIR = Path(__file__).resolve().parent
TARGET_NAME = "IMDB Dataset.csv"


def find_csv(folder: Path):
    matches = list(folder.rglob(TARGET_NAME))
    return matches[0] if matches else None


def main():
    target = DATA_DIR / TARGET_NAME
    if target.exists():
        print(f"Already exists: {target}")
        return

    # First use kagglehub, which manages the download cache for us.
    try:
        import kagglehub

        downloaded = Path(kagglehub.dataset_download(DATASET_SLUG))
        source = find_csv(downloaded)
        if source:
            shutil.copy2(source, target)
            print(f"Downloaded with kagglehub: {target}")
            return
    except Exception as error:
        print(f"kagglehub did not work: {error}")

    # Fallback: use the Kaggle CLI if it is installed and authenticated.
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(DATA_DIR), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        print("Install kaggle or configure kagglehub credentials, then run this script again.", file=sys.stderr)
        raise SystemExit(1)
    except subprocess.CalledProcessError as error:
        print("Kaggle download failed. Check ~/.kaggle/kaggle.json or Kaggle environment variables.", file=sys.stderr)
        raise SystemExit(error.returncode)

    source = find_csv(DATA_DIR)
    if not source:
        raise FileNotFoundError(f"Could not find {TARGET_NAME} after download")
    if source != target:
        shutil.copy2(source, target)
    print(f"Downloaded with Kaggle CLI: {target}")


if __name__ == "__main__":
    main()
