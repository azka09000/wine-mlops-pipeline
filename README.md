# wine-mlops-pipeline

![CI](https://github.com/azka09000/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

A reproducible MLOps pipeline for multi-class wine cultivar classification
(scikit-learn Wine dataset), with MLflow experiment tracking and model registry,
Makefile automation, and a GitHub Actions quality gate.

## Quick start

```bash
make install   # install pinned dependencies
make lint      # flake8 style checks
make test      # unit tests + model quality gate
make train     # train 6 configs, log to MLflow, register the champion
```
