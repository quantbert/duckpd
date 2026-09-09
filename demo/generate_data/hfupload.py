"""Upload the generated feature store to a Hugging Face bucket or dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from build_catalog import build_catalog


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--destination",
        default=os.environ.get("HF_DESTINATION"),
        help=(
            "Target hf://buckets/OWNER/NAME or hf://datasets/OWNER/NAME URI "
            "(default: HF_DESTINATION)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the operation without uploading files.",
    )
    return parser.parse_args()


def parse_destination(destination: str) -> tuple[str, str]:
    """Return the Hub storage type and identifier from a destination URI."""
    for storage_type in ("buckets", "datasets"):
        prefix = f"hf://{storage_type}/"
        if destination.startswith(prefix):
            store_id = destination.removeprefix(prefix).rstrip("/")
            if store_id.count("/") != 1 or any(not part for part in store_id.split("/")):
                break
            return storage_type, store_id
    raise ValueError("destination must be hf://buckets/OWNER/NAME or hf://datasets/OWNER/NAME")


def _uploadable_parquet_files(data_directory: Path) -> list[Path]:
    """Return generated Parquet files while excluding resumable staging data."""
    return [
        path
        for path in data_directory.rglob("*.parquet")
        if not any(part.startswith(".") for part in path.relative_to(data_directory).parts)
    ]


def _upload_exclusions() -> tuple[str, ...]:
    """Return local-only paths that must never be uploaded."""
    return (".cache/**", "**/.cache/**", ".news-staging/**", "**/.news-staging/**")


def _legacy_partition_patterns() -> tuple[str, ...]:
    """Return obsolete coarse partitions removed during a dataset-repo upload."""
    return (
        "ohlcv/year=*/data.parquet",
        "sma/year=*/data.parquet",
        "news/year=*/month=*/data.parquet",
    )


def upload_data(
    data_directory: Path,
    destination: str,
    token: str,
    dry_run: bool = False,
) -> None:
    """Build store metadata and upload a generated dataset to Hugging Face."""
    storage_type, store_id = parse_destination(destination)
    data_directory = data_directory.resolve()
    if not data_directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_directory}")

    catalog = build_catalog(data_directory, store_name=store_id, source=destination)
    parquet_files = _uploadable_parquet_files(data_directory)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {data_directory}")

    print(f"Prepared {len(parquet_files):,} Parquet files from {data_directory}")
    print(f"Catalog: {len(catalog['datasets'])} datasets, {len(catalog['features'])} features")
    print(f"Store README: {data_directory / 'README.md'}")
    print(f"Target: https://huggingface.co/{storage_type}/{store_id} (private)")

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if storage_type == "buckets":
        from huggingface_hub import create_bucket, sync_bucket

        if not dry_run:
            create_bucket(store_id, private=True, exist_ok=True, token=token)
        plan = sync_bucket(
            str(data_directory),
            destination,
            exclude=list(_upload_exclusions()),
            dry_run=dry_run,
            token=token,
        )
        if dry_run:
            print(plan.summary())
        return

    if dry_run:
        print(
            f"Dry run: would upload {len(parquet_files):,} Parquet files and metadata "
            f"to dataset {store_id}"
        )
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(store_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=store_id,
        folder_path=data_directory,
        repo_type="dataset",
        ignore_patterns=list(_upload_exclusions()),
        delete_patterns=list(_legacy_partition_patterns()),
        token=token,
    )


def main() -> None:
    settings = arguments()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")
    if not settings.destination:
        raise ValueError("HF_DESTINATION is not set; pass --destination")
    upload_data(
        data_directory=settings.data,
        destination=settings.destination,
        token=token,
        dry_run=settings.dry_run,
    )


if __name__ == "__main__":
    main()
