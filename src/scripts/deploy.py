from google.cloud import aiplatform  
import os 
from config.config import DeployConfig
import time

config = DeployConfig()
SERVICE_ACCOUNT= os.getenv("SERVICE_ACCOUNT") 
PROJECT_ID = os.getenv("PROJECT_ID")

# 1. Initialize the SDK
# aiplatform.init(
#     project=PROJECT_ID,
#     location=config.LOCATION,
#     staging_bucket=config.BUCKET_NAME,
# )

# # 2. Define the Custom Job
# job = aiplatform.CustomTrainingJob(
#     display_name=config.NAME,
#     script_path=config.SCRIPT_PATH,                  
#     container_uri=config.COTNAINER_URI, 
#     requirements=config.REQUIREMEMTS,    
    
#     # staging_bucket=config.BUCKET_NAME
# )

# # 3. Run the Job
# job.run(
#     machine_type=config.MACHINE_TYPE,
#     accelerator_type='NVIDIA_L4',
#     accelerator_count=config.ACCELERATOR_COUNT,
#     replica_count=config.REPLICA_COUNT,
#     service_account=SERVICE_ACCOUNT,
#     sync=False
# ) 

if not SERVICE_ACCOUNT:
    raise ValueError("SERVICE_ACCOUNT environment variable not set")
if not PROJECT_ID:
    raise ValueError("PROJECT_ID environment variable not set") 
  

# Define a specific output directory for this test 
TIMESTAMP = str(int(time.time()))
SANITY_OUTPUT_DIR = f"{config.BUCKET_NAME}/sanity_checks/test_{TIMESTAMP}"

print(f"Deploying Sanity Check to: {SANITY_OUTPUT_DIR}")

# 1. Initialize the SDK
aiplatform.init(
    project=PROJECT_ID,
    location=config.LOCATION,
    staging_bucket=config.BUCKET_NAME,
)

# 2. Define the Custom Job (Pointing to the SANITY script)
job = aiplatform.CustomTrainingJob(
    display_name="sanity-check-gcs-write",
    script_path="sanity.py",  # <--- CHANGED: Points to the test script
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest",
    requirements=["google-cloud-storage"], # Ensure storage lib is present
    staging_bucket=config.BUCKET_NAME
)

# 3. Run the Job (CPU ONLY)
job.run(
    # machine_type=config.MACHINE_TYPE, # <--- COMMENTED OUT
    machine_type="n1-standard-4",       # <--- ADDED: Cheap CPU machine
    
    # accelerator_type='NVIDIA_L4',     # <--- COMMENTED OUT
    accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",              # <--- ADDED: No GPU
    
    # accelerator_count=config.ACCELERATOR_COUNT, # <--- COMMENTED OUT
    accelerator_count=0,                # <--- ADDED: Zero accelerators
    
    replica_count=1,
    service_account=SERVICE_ACCOUNT,
    
    # CRITICAL: This sets AIP_MODEL_DIR in the container
    base_output_dir=SANITY_OUTPUT_DIR,  
    
    sync=False # Set to True so you see logs in your terminal immediately
)