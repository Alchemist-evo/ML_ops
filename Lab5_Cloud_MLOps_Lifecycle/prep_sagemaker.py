"""Package the exported champion for SageMaker and upload to S3."""
import tarfile, sagemaker, pathlib, shutil

# reuse Lab 2's export:  python export_champion.py  first
src = pathlib.Path("model_export/model.pkl")
shutil.copy(src, "model.joblib")          # container convention

with tarfile.open("model.tar.gz", "w:gz") as t:
    t.add("model.joblib")

sess = sagemaker.Session()
bucket = sess.default_bucket()
s3_uri = sess.upload_data("model.tar.gz", bucket, "cartvista/model")
print("Uploaded:", s3_uri)
pathlib.Path("s3_uri.txt").write_text(s3_uri)   # read by deploy_endpoint.py
