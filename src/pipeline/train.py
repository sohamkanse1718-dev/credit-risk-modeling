"""CLI entry point: ``python -m src.pipeline.train``."""

from pathlib import Path

from .training_pipeline import train


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    metadata = train(root / "data" / "processed", root / "models")
    print(
        f"Saved models/model.pkl and models/model_metadata.json "
        f"(validation ROC-AUC: {metadata['validation_metrics']['roc_auc']:.4f})."
    )


if __name__ == "__main__":
    main()
