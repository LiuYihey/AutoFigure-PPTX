from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
import os

def download_models():
    # Set the directory where models will be saved
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Download SAM3 model from ModelScope
    print("\nDownloading SAM3 model from ModelScope (facebook/sam3)...")
    try:
        sam3_path = os.path.join(models_dir, "sam3")
        ms_snapshot_download(model_id="facebook/sam3", local_dir=sam3_path)
        print(f"Successfully downloaded SAM3 to {sam3_path}")
    except Exception as e:
        print(f"Failed to download SAM3 from ModelScope: {e}")

    print(f"\nDownloading models to {models_dir}...")

    # 2. Download RMBG-2.0 model from ModelScope
    print("\nDownloading RMBG-2.0 model from ModelScope (briaai/RMBG-2.0)...")
    try:
        rmbg_path = os.path.join(models_dir, "RMBG-2.0")
        ms_snapshot_download(model_id="briaai/RMBG-2.0", local_dir=rmbg_path)
        print(f"Successfully downloaded RMBG-2.0 to {rmbg_path}")
    except Exception as e:
        print(f"Failed to download RMBG-2.0 from ModelScope: {e}")

    print("\nModel download process completed.")
    print(f"SAM3 Checkpoint Path: {os.path.join(models_dir, 'sam3', 'sam3.pt')}")
    print(f"RMBG Model Path: {os.path.join(models_dir, 'RMBG-2.0')}")

if __name__ == "__main__":
    download_models()
