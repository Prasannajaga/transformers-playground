import argparse
import logging
import os
import sys
import time
from pathlib import Path

DEFAULT_BUCKET = "gs://transformer-garage"
DEFAULT_LOCATION = "us-central1"
DEFAULT_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
DEFAULT_REQUIREMENTS: list[str] = []
DEFAULT_DISPLAY_NAME = "model-training-job"
DEFAULT_MACHINE_TYPE = "g2-standard-4"
DEFAULT_ACCELERATOR_TYPE = "NVIDIA_L4"
DEFAULT_ACCELERATOR_COUNT = 1
DEFAULT_REPLICA_COUNT = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Vertex AI custom training job")

    parser.add_argument("--project-id", default=os.getenv("PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--service-account", default=os.getenv("SERVICE_ACCOUNT"), help="Service account email")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="Vertex region")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="Staging/output GCS bucket")

    parser.add_argument("--script-path", required=True, help="Path to training entry script")
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME, help="Training job display name")
    parser.add_argument("--container-uri", default=DEFAULT_CONTAINER_URI, help="Training container URI")
    parser.add_argument("--requirement", action="append", default=None, help="Repeatable pip requirement")

    parser.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE, help="Worker machine type")
    parser.add_argument("--accelerator-type", default=DEFAULT_ACCELERATOR_TYPE, help="Accelerator type")
    parser.add_argument("--accelerator-count", type=int, default=DEFAULT_ACCELERATOR_COUNT, help="Number of accelerators")
    parser.add_argument("--replica-count", type=int, default=DEFAULT_REPLICA_COUNT, help="Replica count")

    parser.add_argument("--base-output-dir", default=None, help="Explicit base output GCS dir")
    parser.add_argument("--sync", default=False, action="store_true", help="Wait until job completion")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    return parser.parse_args()


def build_base_output_dir(bucket: str, display_name: str, explicit: str | None) -> str:
    if explicit:
        return explicit.rstrip("/")

    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in display_name).strip("-")
    if not safe_name:
        safe_name = "training-job"

    return f"{bucket.rstrip('/')}/vertex_training/{safe_name}/{int(time.time())}"


def run() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if not args.project_id:
        raise ValueError("PROJECT_ID is required (env PROJECT_ID or --project-id)")
    if not args.service_account:
        raise ValueError("SERVICE_ACCOUNT is required (env SERVICE_ACCOUNT or --service-account)")
    if not args.bucket.startswith("gs://"):
        raise ValueError(f"Bucket must start with gs://, got: {args.bucket}")

    script_path = Path(args.script_path).expanduser().resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    requirements = args.requirement if args.requirement is not None else list(DEFAULT_REQUIREMENTS)
    base_output_dir = build_base_output_dir(args.bucket, args.display_name, args.base_output_dir)

    from google.cloud import aiplatform

    logger.info("Initializing Vertex AI")
    aiplatform.init(project=args.project_id, location=args.location, staging_bucket=args.bucket)

    logger.info("Creating custom training job")
    job = aiplatform.CustomTrainingJob(
        display_name=args.display_name,
        script_path=str(script_path),
        container_uri=args.container_uri,
        requirements=requirements,
        staging_bucket=args.bucket, 
    )

    run_kwargs = {
        "machine_type": args.machine_type,
        "accelerator_type": args.accelerator_type,
        "accelerator_count": int(args.accelerator_count),
        "replica_count": args.replica_count,
        "service_account": args.service_account,
        "base_output_dir": base_output_dir,
        "sync": args.sync,
    }

    logger.info("Submitting training job")
    logger.info("display_name=%s", args.display_name)
    logger.info("script_path=%s", script_path)
    logger.info("base_output_dir=%s", base_output_dir)

    job.run(**run_kwargs)

    logger.info("Job submitted successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        logging.getLogger(__name__).exception("Deployment failed: %s", exc)
        sys.exit(1)
