from google.cloud import aiplatform  
import os 
from config.config import DeployConfig

config = DeployConfig()
SERVICE_ACCOUNT= os.getenv("SERVICE_ACCOUNT") 
PROJECT_ID = os.getenv("PROJECT_ID")

# 1. Initialize the SDK
aiplatform.init(
    project=PROJECT_ID,
    location=config.LOCATION,
    staging_bucket=config.BUCKET_NAME
)

# 2. Define the Custom Job
job = aiplatform.CustomTrainingJob(
    display_name=config.NAME,
    script_path=config.SCRIPT_PATH,                  
    container_uri=config.COTNAINER_URI, 
    requirements=config.REQUIREMEMTS,    
    
    # staging_bucket=config.BUCKET_NAME
)

# 3. Run the Job
job.run(
    machine_type=config.MACHINE_TYPE,
    accelerator_type='NVIDIA_L4',
    accelerator_count=config.ACCELERATOR_COUNT,
    replica_count=config.REPLICA_COUNT,
    service_account=SERVICE_ACCOUNT,
    sync=False
) 