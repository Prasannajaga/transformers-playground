# sanity_check.py
import os
import time
from google.cloud import storage

def test_gcs_write():
    # Vertex AI automatically sets this env var if you pass 'base_output_dir' in the job config
    # It will look like: gs://your-bucket/sanity-check-job-name/model
    model_dir = os.environ.get('AIP_MODEL_DIR')
    
    print(f"Checking access to AIP_MODEL_DIR: {model_dir}")

    if not model_dir:
        raise ValueError("AIP_MODEL_DIR not found. Did you set 'base_output_dir' in the deploy script?")

    if not model_dir.startswith("gs://"):
        raise ValueError(f"AIP_MODEL_DIR must be a GCS path, got: {model_dir}")

    # Parse bucket and blob
    # format: gs://bucket-name/path/to/model
    parts = model_dir[5:].split("/")
    bucket_name = parts[0]
    prefix = "/".join(parts[1:])
    
    print(f"Target Bucket: {bucket_name}")
    print(f"Target Prefix: {prefix}")

    # Create a dummy file locally
    local_file = "success_marker.txt"
    with open(local_file, "w") as f:
        f.write(f"Write test successful at {time.ctime()}")

    # Upload to GCS
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{prefix}/{local_file}")
        blob.upload_from_filename(local_file)
        
        print("\n" + "="*50)
        print(f"SUCCESS: File uploaded to {model_dir}/{local_file}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Could not write to GCS: {e}")
        raise e

if __name__ == "__main__":
    test_gcs_write()