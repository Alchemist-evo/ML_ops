import os, pathlib
from sagemaker.sklearn import SKLearnModel
import sagemaker

# <s3_uri from Step 2> is written to s3_uri.txt by prep_sagemaker.py;
# <your SageMaker execution role ARN> comes from the SAGEMAKER_ROLE_ARN env var.
S3_URI = os.environ.get("S3_MODEL_URI") or pathlib.Path("s3_uri.txt").read_text().strip()
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]

model = SKLearnModel(
    model_data=S3_URI,
    role=ROLE_ARN,
    entry_point="inference.py",
    framework_version=os.environ.get("SKLEARN_FRAMEWORK_VERSION", "1.2-1"),
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",        # smallest -- do not change up
    endpoint_name="cartvista-churn-ep",
)
print("Endpoint ready:", predictor.endpoint_name)
