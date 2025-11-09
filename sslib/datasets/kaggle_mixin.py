"""Simplified mixin for downloading datasets from Kaggle."""

import zipfile
import shutil
import requests
from pathlib import Path
from typing import Optional
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)


class KaggleDatasetMixin:
    """
    Simplified mixin for Kaggle dataset downloads.

    Subclasses must implement:
    - _get_kaggle_dataset_id() -> str
    - _verify_structure() -> bool
    """

    @abstractmethod
    def _get_kaggle_dataset_id(self) -> str:
        """Return Kaggle dataset ID (e.g., 'username/dataset-name')."""
        pass

    @abstractmethod
    def _verify_structure(self) -> bool:
        """Verify dataset structure is correct after extraction."""
        pass

    def _download_from_kaggle(
        self,
        dataset_id: Optional[str] = None,
        zip_filename: Optional[str] = None,
    ) -> None:
        """
        Download and extract dataset from Kaggle.

        Simplified flow:
        1. Download zip file
        2. Extract to root
        3. Verify structure
        4. Clean up
        """
        dataset_id = dataset_id or self._get_kaggle_dataset_id()

        # Generate zip filename
        if zip_filename is None:
            zip_filename = f"{dataset_id.split('/')[-1]}.zip"

        # Ensure root directory exists
        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / zip_filename

        try:
            # Step 1: Download
            logger.info(f"Downloading {dataset_id} from Kaggle...")
            print(f"Downloading {dataset_id} from Kaggle...")
            self._download_file(dataset_id, zip_path)

            # Step 2: Extract
            logger.info("Extracting dataset...")
            print("Extracting dataset...")
            self._extract_zip(zip_path)

            # Step 3: Verify
            if not self._verify_structure():
                raise RuntimeError("Dataset structure verification failed")

            # Step 4: Clean up
            logger.info("Cleaning up...")
            print("Cleaning up...")
            zip_path.unlink()

            logger.info("Dataset download completed successfully")
            print("Dataset download completed successfully")

        except requests.exceptions.RequestException as e:
            self._handle_download_error(e, dataset_id)
            raise

        except zipfile.BadZipFile as e:
            self._handle_zip_error(e, zip_path)
            raise

        except Exception as e:
            self._cleanup_on_error(zip_path)
            logger.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"Dataset download failed: {str(e)}") from e

    def _download_file(self, dataset_id: str, output_path: Path) -> None:
        """Download file from Kaggle API."""
        kaggle_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"

        response = requests.get(kaggle_url, stream=True)
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
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                print()  # New line after progress
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        logger.info(f"Download completed: {output_path}")

    def _extract_zip(self, zip_path: Path) -> None:
        """Extract zip file to root directory."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

    def _handle_download_error(self, error: Exception, dataset_id: str) -> None:
        """Handle download errors with helpful messages."""
        logger.error(f"Download failed: {str(error)}")
        print(f"\n❌ Error downloading dataset: {str(error)}")
        print("\n📝 Please check:")
        print("   1. Internet connection")
        print("   2. Kaggle API credentials (~/.kaggle/kaggle.json)")
        print(f"   3. Dataset access: https://www.kaggle.com/datasets/{dataset_id}")
        print("\n💡 Manual download:")
        print(f"   1. Visit: https://www.kaggle.com/datasets/{dataset_id}")
        print(f"   2. Download to: {self.root}")
        print("   3. Extract the files")

    def _handle_zip_error(self, error: Exception, zip_path: Path) -> None:
        """Handle zip extraction errors."""
        logger.error(f"Zip extraction failed: {str(error)}")
        print(f"\n❌ Error extracting zip file: {str(error)}")

        if zip_path.exists():
            zip_path.unlink()
            print("Removed corrupted zip file")

    def _cleanup_on_error(self, zip_path: Path) -> None:
        """Clean up files after error."""
        if zip_path.exists():
            try:
                zip_path.unlink()
                logger.info("Cleaned up partial download")
            except Exception as e:
                logger.warning(f"Failed to clean up: {str(e)}")

    def _find_file(self, filename: str) -> Optional[Path]:
        """
        Find a file anywhere in root directory.
        Returns first match or None.
        """
        for path in self.root.rglob(filename):
            if path.is_file():
                return path
        return None

    def _find_directory(self, dirname: str) -> Optional[Path]:
        """
        Find a directory anywhere in root directory.
        Returns first match or None.
        """
        for path in self.root.rglob(dirname):
            if path.is_dir():
                return path
        return None

    def _move_to_root(self, source: Path, target_name: str) -> Path:
        """Move file or directory to root with new name."""
        target = self.root / target_name

        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        shutil.move(str(source), str(target))
        logger.info(f"Moved {source.name} to {target_name}")

        return target
