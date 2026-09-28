# ML Ops Labs

This repository contains eight hands-on MLOps labs. Each lab has a `run_labN.sh`
script that runs its workflow and, where applicable, installs its Python
requirements when `INSTALL=1` is set.

## Prerequisites

- Bash and Python 3.10 or newer
- Git and `curl`
- Docker with the Compose plugin for Labs 2 and 4
- Ollama for Labs 6–8
- AWS CLI credentials and a SageMaker execution role for Lab 5

Run commands from the repository root. The scripts change into their own lab
directories, so generated files and local databases stay with the corresponding
lab.

## Running a lab

Install a lab's Python dependencies and run it:

```bash
INSTALL=1 bash Lab1_MLflow_Experiment_Tracking/run_lab1.sh
```

After dependencies are installed, you can omit `INSTALL=1` on later runs:

```bash
bash Lab1_MLflow_Experiment_Tracking/run_lab1.sh
```

The scripts use the active Python interpreter by default. Set `PYTHON` to use a
specific interpreter. They also activate a lab-local `.venv` if one exists:

```bash
python3 -m venv Lab1_MLflow_Experiment_Tracking/.venv
INSTALL=1 PYTHON=python3 bash Lab1_MLflow_Experiment_Tracking/run_lab1.sh
```

## Labs

Run the scripts in order for the recommended progression:

| Lab | Command | Focus |
|---|---|---|
| 1 | `bash Lab1_MLflow_Experiment_Tracking/run_lab1.sh` | MLflow experiments and model registry |
| 2 | `bash Lab2_Docker_Airflow_Pipeline/run_lab2.sh` | Docker scoring and Airflow orchestration |
| 3 | `bash Lab3_Model_Serving_Microservice/run_lab3.sh` | FastAPI model serving |
| 4 | `bash Lab4_Monitoring_Drift_AB_Testing/run_lab4.sh` | Prometheus, Grafana, drift and A/B testing |
| 5 | `bash Lab5_Cloud_MLOps_Lifecycle/run_lab5.sh` | AWS SageMaker deployment and model registry |
| 6 | `bash Lab6_RAG_VectorDB_Evaluation/run_lab6.sh` | RAG, local Qdrant and evaluation |
| 7 | `bash Lab7_vLLM_Observability_CICD/run_lab7.sh` | LLM benchmarking, Langfuse and CI gates |
| 8 | `bash Lab8_Agentic_Workflow_Tracing/run_lab8.sh` | Agent workflow and evaluation |

To install each lab's dependencies on its first run, prefix its command with
`INSTALL=1`, for example:

```bash
INSTALL=1 bash Lab2_Docker_Airflow_Pipeline/run_lab2.sh
```

Some labs run local services or take several minutes. Read the output and
prerequisites below before starting them.

## Lab-specific setup

- **Lab 2:** Docker must be running. Airflow is installed by the script if it
  is not present; Airflow should be run under WSL2 on Windows. Set
  `SKIP_DOCKER=1` or `SKIP_AIRFLOW=1` to skip those sections.
- **Lab 4:** Docker Compose starts the Prometheus and Grafana stack. The script
  stops the stack when it exits; set `KEEP_STACK=1` to leave it running.
  `MINUTES` and `MINUTES_AB` control traffic-generation duration.
- **Lab 5:** This lab creates billable AWS SageMaker resources. Configure the
  AWS CLI in the intended region and set `SAGEMAKER_ROLE_ARN` before running.
  The script attempts endpoint teardown on exit, but AWS charges may still
  apply; check the AWS console to confirm resources are removed.
- **Labs 6–8:** Start Ollama before running. The scripts use
  `OLLAMA_MODEL=llama3.2:1b` by default and pull it if it is missing. Lab 6
  also uses `EMBEDDER_MODEL=all-MiniLM-L6-v2`.
- **Labs 7–8:** To record traces, provide a reachable Langfuse instance and
  export its credentials. Without keys, Lab 7 skips tracing and Lab 8 still
  runs without recording traces.

Example Ollama setup:

```bash
ollama serve
# In another terminal, if needed:
ollama pull llama3.2:1b
```

Langfuse tracing environment:

```bash
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=your-public-key
export LANGFUSE_SECRET_KEY=your-secret-key
```

## Useful options

Options are environment variables and can be set for a single run:

```bash
PORT=8001 bash Lab3_Model_Serving_Microservice/run_lab3.sh
ROUTE=cpu bash Lab7_vLLM_Observability_CICD/run_lab7.sh
OLLAMA_MODEL=llama3.2:1b bash Lab8_Agentic_Workflow_Tracing/run_lab8.sh
```

For each lab's other options and workflow details, see the comments at the top
of its `run_labN.sh` script.
