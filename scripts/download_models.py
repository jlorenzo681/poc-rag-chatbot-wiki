import os
import urllib.request
from pathlib import Path

def download_fasttext_model():
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    output_dir = Path("data/models")
    output_path = output_dir / "lid.176.ftz"

    if output_path.exists():
        print(f"✓ Model already exists at {output_path}")
        return

    print(f"⬇️  Downloading FastText model to {output_path}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(model_url, output_path)
        print("✓ Download complete!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        exit(1)

if __name__ == "__main__":
    download_fasttext_model()
