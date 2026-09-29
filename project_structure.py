from pathlib import Path


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------
# Folder structure
# ---------------------------------------------------------

DIRECTORIES = [
    # Data
    "data/raw",
    "data/interim",
    "data/processed",

    # Notebooks
    "notebooks",

    # Source code
    "src/config",
    "src/data",
    "src/features",
    "src/models",
    "src/pipeline",
    "src/utils",

    # Tests
    "tests",

    # Model artifacts
    "models",

    # Reports
    "reports/figures",
    "reports/model_results",

    # Configuration
    "configs",
]


# ---------------------------------------------------------
# Files to create
# ---------------------------------------------------------

FILES = [
    # Source package
    "src/__init__.py",

    # Config
    "src/config/__init__.py",
    "src/config/config.py",

    # Data
    "src/data/__init__.py",
    "src/data/ingestion.py",
    "src/data/validation.py",

    # Features
    "src/features/__init__.py",
    "src/features/engineering.py",

    # Models
    "src/models/__init__.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/models/predict.py",

    # Pipeline
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",

    # Utils
    "src/utils/__init__.py",
    "src/utils/common.py",

    # Tests
    "tests/__init__.py",
    "tests/test_data.py",
    "tests/test_features.py",
    "tests/test_model.py",

    # Configuration
    "configs/config.yaml",

    # Requirements
    "requirements.txt",

    # Keep empty directories in Git
    "models/.gitkeep",
]


# ---------------------------------------------------------
# Optional notebook files
# ---------------------------------------------------------

NOTEBOOKS = [
    "notebooks/01_data_understanding.ipynb",
    "notebooks/02_eda.ipynb",
    "notebooks/03_feature_engineering.ipynb",
    "notebooks/04_model_experimentation.ipynb",
]


# ---------------------------------------------------------
# Create directories
# ---------------------------------------------------------

def create_directories():
    for directory in DIRECTORIES:
        path = ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {path.relative_to(ROOT)}")


# ---------------------------------------------------------
# Create files
# ---------------------------------------------------------

def create_files():

    for file in FILES:
        path = ROOT / file

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            print(f"[FILE] {path.relative_to(ROOT)}")
        else:
            print(f"[SKIP] {path.relative_to(ROOT)} already exists")


# ---------------------------------------------------------
# Create notebooks
# ---------------------------------------------------------

def create_notebooks():

    for notebook in NOTEBOOKS:
        path = ROOT / notebook

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(
                """{
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}
""",
                encoding="utf-8"
            )

            print(f"[FILE] {path.relative_to(ROOT)}")

        else:
            print(f"[SKIP] {path.relative_to(ROOT)} already exists")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\nCreating ML project structure...\n")

    create_directories()
    create_files()
    create_notebooks()

    print("\nProject structure created successfully.")


if __name__ == "__main__":
    main()