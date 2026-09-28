import boto3, json

rt = boto3.client("sagemaker-runtime")
at_risk = [[2, 2500.0, 1, 6, 0, 2]]
loyal   = [[60, 1800.0, 8, 0, 1, 20]]

for name, row in (("at_risk", at_risk), ("loyal", loyal)):
    r = rt.invoke_endpoint(
        EndpointName="cartvista-churn-ep",
        ContentType="application/json",
        Body=json.dumps(row))
    print(name, "->", r["Body"].read().decode())
