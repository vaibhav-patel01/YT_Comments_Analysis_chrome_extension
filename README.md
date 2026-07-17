# YouTube Comment Sentiment Analysis — End-to-End MLOps Project

A production-style sentiment analysis system for YouTube comments, covering the full lifecycle from data ingestion to a deployed, auto-scaled inference API — with experiment tracking, pipeline versioning, and full CI/CD automation.

to use the extension visit - https://github.com/vaibhav-patel01/Sentiment_Reel/blob/main/README.md

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Data Source](#data-source)
4. [Data Cleaning & NLP Preprocessing](#data-cleaning--nlp-preprocessing)
5. [Feature Engineering (TF-IDF)](#feature-engineering-tf-idf)
6. [Experiment Tracking (MLflow + DagsHub)](#experiment-tracking-mlflow--dagshub)
7. [Model Building & Results](#model-building--results)
8. [DVC Pipeline](#dvc-pipeline)
9. [Model Registry](#model-registry)
10. [Backend (FastAPI)](#backend-fastapi)
11. [Frontend (Chrome Extension)](#frontend-chrome-extension)
12. [Dockerization](#dockerization)
13. [Deployment (AWS ASG + CodeDeploy)](#deployment-aws-asg--codedeploy)
14. [CI/CD Pipeline](#cicd-pipeline)
15. [Project Structure](#project-structure)
16. [Known Limitations](#known-limitations)

---

## Project Overview

This project classifies the sentiment of YouTube comments (positive / negative / neutral) and ships the model as a real-world product: a Chrome extension that lets users see sentiment analysis for comments directly on YouTube, backed by a FastAPI service running on AWS.

**Pipeline at a glance:**

```
HuggingFace Dataset → Cleaning/NLP → TF-IDF → MLflow Experiments (DagsHub)
        → Best Model (LightGBM) → DVC Pipeline → Model Registry
        → FastAPI Backend → Docker → CI/CD → AWS Auto Scaling Group
        → Chrome Extension (Frontend)
```

---

## Architecture

```
 ┌────────────┐     ┌───────────────┐     ┌────────────────┐
 │ HuggingFace│ --> │ Data Cleaning │ --> │ Feature Eng.   │
 │  Dataset   │     │  (NLP/Regex)  │     │ (TF-IDF)       │
 └────────────┘     └───────────────┘     └───────┬────────┘
                                                   │
                                                   v
                                        ┌─────────────────────┐
                                        │  MLflow + DagsHub    │
                                        │  Experiment Tracking │
                                        └──────────┬──────────┘
                                                   │ best model
                                                   v
                          ┌───────────────────────────────────────┐
                          │  DVC Pipeline (data→features→model)   │
                          └───────────────────┬───────────────────┘
                                              │
                                              v
                                  ┌───────────────────────┐
                                  │   Model Registry       │
                                  │   (via CI/CD)          │
                                  └───────────┬───────────┘
                                              │
                                              v
                     ┌─────────────┐   ┌─────────────┐   ┌───────────────┐
                     │  FastAPI    │-->│   Docker    │-->│  AWS ECR      │
                     │  Backend    │   │   Image     │   │               │
                     └─────────────┘   └─────────────┘   └───────┬───────┘
                                                                  │
                                                                  v
                                                     ┌─────────────────────┐
                                                     │  AWS CodeDeploy +   │
                                                     │  Auto Scaling Group │
                                                     └──────────┬──────────┘
                                                                │
                                                                v
                                                    ┌────────────────────┐
                                                    │  Chrome Extension  │
                                                    │  (Frontend)        │
                                                    └────────────────────┘
```

---

## Data Source

The raw comment data is sourced from the HuggingFace dataset:

**[AmaanP314/youtube-comment-sentiment](https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment)**

Ingestion is handled by `src/data/data_ingestion.py`, which pulls the raw data and splits it into train/test sets (`test_size: 0.2`, configured via `params.yaml`).

---

## Data Cleaning & NLP Preprocessing

Handled in `src/data/data_preprocessing.py`. Key steps:

- Unicode normalization and emoji/multilingual text handling
- Noise removal: URLs, HTML tags, special characters
- Case normalization, stopword handling, and tokenization
- Custom text-cleaning utilities built around `unicodedata` and `str.translate()` for robust handling of non-ASCII and emoji-heavy comments

This stage converts raw, noisy YouTube comments into a clean text corpus suitable for vectorization.

---

## Feature Engineering (TF-IDF)

Handled in `src/features/feature_engineering.py`, configured via `params.yaml`:

| Parameter | Value |
|---|---|
| `ngram_range` | (1, 3) |
| `max_features` | 5000 |

The fitted vectorizer is persisted as `tfidf_vectorizer.pkl` so the exact same transformation is used at inference time.

---

## Experiment Tracking (MLflow + DagsHub)

MLflow is connected to a **DagsHub** remote tracking server, allowing every experiment run (parameters, metrics, artifacts) to be logged and compared centrally.

Multiple algorithms and hyperparameter configurations were tried and tracked, including Logistic Regression, SVM, and gradient-boosted trees, before settling on the final model.

---

## Model Building & Results

Final model: **LightGBM (LGBM) Classifier**, configured via `params.yaml`:

| Parameter | Value |
|---|---|
| `learning_rate` | 0.09611 |
| `max_depth` | 15 |
| `n_estimators` | 417 |
| `class_weight` | balanced |
| `objective` | multiclass |
| `metric` | multi_logloss |

**Result: ~86% accuracy.**

The main source of remaining error is **sarcastic comments** — sarcasm inverts sentiment polarity in ways that are hard for a TF-IDF + tree-based model to capture, since it relies on lexical/textual features rather than deeper contextual or pragmatic understanding.

Evaluation logic lives in `src/model/model_evaluation.py`, which produces `experiment_info.json` capturing the run's metrics and metadata for downstream registration.

---

## DVC Pipeline

The full pipeline is version-controlled and reproducible via **DVC** (`dvc.yaml`), with 6 stages:

1. **data_ingestion** — pulls raw data, applies train/test split
2. **data_preprocessing** — cleans and normalizes text
3. **feature_engineering** — fits/applies TF-IDF vectorization
4. **model_building** — trains the LightGBM model
5. **model_evaluation** — evaluates the model, logs to MLflow, writes `experiment_info.json`
6. **register_model** — registers the model in the model registry, writes `registration_status.log`

Run the full pipeline with:

```bash
dvc repro
```

Push data/model artifacts to remote storage with:

```bash
dvc push
```

---

## Model Registry

`src/model/register_model.py` promotes the best-performing model (based on `experiment_info.json`) into the **MLflow Model Registry** on DagsHub. This step is automated as part of CI/CD, so every successful pipeline run on the main branch can push a new model version without manual intervention.

---

## Backend (FastAPI)

The trained model and TF-IDF vectorizer are served behind a **FastAPI** application, exposing endpoints for:

- Single comment sentiment prediction
- Batch comment sentiment prediction (for use by the Chrome extension against a full comments thread)

FastAPI was chosen for its speed, automatic OpenAPI docs, and native async support suited for I/O-bound serving.

---

## Frontend (Chrome Extension)

A Chrome extension serves as the user-facing frontend. It:

- Reads comments from the currently open YouTube video page
- Sends them to the FastAPI backend for sentiment prediction
- Displays sentiment results (e.g., positive/negative/neutral breakdown) directly in the YouTube UI

Since the extension isn't published on the Chrome Web Store, it's loaded manually via Chrome's developer mode (`chrome://extensions` → **Load unpacked**).

---

## Dockerization

The FastAPI backend is containerized via the project's `Dockerfile`, producing a self-contained image with the model, vectorizer, and API service. This image is what gets pushed to AWS ECR and deployed.

Build locally:

```bash
docker build -t vaibhavpatel01project1 .
docker run -p 8000:8000 vaibhavpatel01project1
```

---

## Deployment (AWS ASG + CodeDeploy)

Deployment infrastructure:

- **AWS ECR** — stores the built Docker image
- **AWS CodeDeploy** — orchestrates deployment to instances using `appspec.yml`
- **AWS Auto Scaling Group (ASG)** — runs the containerized backend across scalable EC2 instances behind a load balancer

`appspec.yml` defines the deployment lifecycle hooks:

| Hook | Script | Purpose |
|---|---|---|
| `BeforeInstall` | `deploy/scripts/install_dependencies.sh` | Prepares the instance (e.g., Docker install) |
| `ApplicationStart` | `deploy/scripts/start_docker.sh` | Pulls and runs the latest container |

---

## CI/CD Pipeline

The entire project — from data pipeline to deployment — is automated via a GitHub Actions workflow (`cicd.yaml`), triggered on every `push`:

1. **Setup** — checkout code, set up Python 3.13, cache pip dependencies
2. **Run DVC pipeline** — `dvc repro` (using DagsHub + AWS credentials from secrets)
3. **Push DVC artifacts** — `dvc push` to remote storage
4. **Auto-commit** — commits and pushes any pipeline-generated changes back to the repo
5. **Model test** — runs `pytest scripts/model_test.py` to verify the model loads correctly
6. **Build & push Docker image** — logs in to AWS ECR, builds and pushes the image
7. **Package deployment bundle** — zips `appspec.yml` + deploy scripts
8. **Upload to S3** — uploads the deployment bundle
9. **Deploy** — triggers an AWS CodeDeploy deployment to the target deployment group, rolling out to the Auto Scaling Group

This means a single `git push` takes the project from raw data through a freshly trained/evaluated/registered model to a live deployment — no manual steps required.

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
├── src/
│   ├── data/
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── model/
│       ├── model_building.py
│       ├── model_evaluation.py
│       └── register_model.py
├── deploy/
│   └── scripts/
│       ├── install_dependencies.sh
│       └── start_docker.sh
├── extension/                 # Chrome extension frontend
├── scripts/
│   └── model_test.py
├── tfidf_vectorizer.pkl
├── model/
│   └── model.pkl
├── dvc.yaml
├── params.yaml
├── appspec.yml
├── Dockerfile
├── .github/workflows/cicd.yaml
└── README.md
```

---

## Known Limitations

- **Sarcasm detection**: the primary source of misclassification (~14% error rate) — a TF-IDF + LightGBM setup relies on lexical patterns and doesn't capture the contextual/tonal cues sarcasm depends on.
- **Multilingual coverage**: while preprocessing handles Unicode/emoji, deeper multilingual sentiment nuance is limited by TF-IDF's bag-of-words nature.
- Potential future improvements: transformer-based embeddings (e.g., BERT-family models) or aspect-based sentiment analysis for better handling of sarcasm and context.


a chrome extension for YT comments analysis

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
