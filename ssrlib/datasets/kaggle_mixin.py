"""Mixin for downloading datasets from Kaggle.

The mixin assumes the consuming class provides:
    - ``self.root`` : Path to the dataset root directory.
    - ``_verify_structure() -> bool`` : Validate post-extraction layout.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class KaggleDatasetMixin:
    """Shared zip-download / extract / locate logic for Kaggle datasets."""

    @abstractmethod
    def _get_kaggle_dataset_id(self) -> str:
        """Return Kaggle dataset ID (e.g., 'username/dataset-name')."""

    @abstractmethod
    def _verify_structure(self) -> bool:
        """Verify dataset structure is correct after extraction."""

    def _download_from_kaggle(
        self,
        dataset_id: Optional[str] = None,
        zip_filename: Optional[str] = None,
    ) -> None:
        dataset_id = dataset_id or self._get_kaggle_dataset_id()
        if zip_filename is None:
            zip_filename = f"{dataset_id.split('/')[-1]}.zip"

        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / zip_filename

        try:
            logger.info("Downloading %s from Kaggle...", dataset_id)
            self._download_file(dataset_id, zip_path)

            logger.info("Extracting dataset...")
            self._extract_zip(zip_path)

            if not self._verify_structure():
                raise RuntimeError("Dataset structure verification failed")

            logger.info("Cleaning up zip file")
            zip_path.unlink()
            logger.info("Dataset download completed")

        except requests.exceptions.RequestException as exc:
            self._handle_download_error(exc, dataset_id)
            raise
        except zipfile.BadZipFile as exc:
            self._handle_zip_error(exc, zip_path)
            raise
        except Exception as exc:
            self._cleanup_on_error(zip_path)
            logger.error("Unexpected error: %s", exc)
            raise RuntimeError(f"Dataset download failed: {exc}") from exc

    def _download_file(self, dataset_id: str, output_path: Path) -> None:
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            if total_size > 0:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        # Single-line progress; users wanting silent mode can
                        # raise the logger level for this module.
                        logger.debug("Progress: %.1f%%", progress)
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info("Download completed: %s", output_path)

    def _extract_zip(self, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

    def _handle_download_error(self, error: Exception, dataset_id: str) -> None:
        logger.error(
            "Download failed: %s. Check internet, ~/.kaggle/kaggle.json, and "
            "dataset access at https://www.kaggle.com/datasets/%s",
            error,
            dataset_id,
        )

    def _handle_zip_error(self, error: Exception, zip_path: Path) -> None:
        logger.error("Zip extraction failed: %s", error)
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Removed corrupted zip file")

    def _cleanup_on_error(self, zip_path: Path) -> None:
        if zip_path.exists():
            try:
                zip_path.unlink()
                logger.info("Cleaned up partial download")
            except Exception as exc:
                logger.warning("Failed to clean up: %s", exc)

    def _find_file(self, filename: str) -> Optional[Path]:
        for path in self.root.rglob(filename):
            if path.is_file():
                return path
        return None

    def _find_directory(self, dirname: str) -> Optional[Path]:
        for path in self.root.rglob(dirname):
            if path.is_dir():
                return path
        return None

    def _move_to_root(self, source: Path, target_name: str) -> Path:
        target = self.root / target_name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(source), str(target))
        logger.info("Moved %s to %s", source.name, target_name)
        return target
