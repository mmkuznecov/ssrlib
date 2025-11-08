"""Mixin for downloading datasets from Kaggle."""

import zipfile
import shutil
import requests
from pathlib import Path
from typing import Optional
from abc import abstractmethod


class KaggleDatasetMixin:
    """Mixin providing Kaggle download functionality for datasets.

    Classes using this mixin should define:
    - self.root: Path to dataset root directory
    - _get_kaggle_dataset_id(): Return the Kaggle dataset ID
    - _get_manual_download_instructions(): Return manual download instructions
    - _organize_extracted_files(): Organize files after extraction (optional)
    """

    @abstractmethod
    def _get_kaggle_dataset_id(self) -> str:
        """Get the Kaggle dataset identifier (e.g., 'username/dataset-name').

        Returns:
            Kaggle dataset ID
        """
        pass

    @abstractmethod
    def _get_manual_download_instructions(self) -> list[str]:
        """Get manual download instructions for this dataset.

        Returns:
            List of instruction strings to display to user
        """
        pass

    def _organize_extracted_files(self) -> None:
        """Organize extracted files into expected structure.

        Override this method in subclass if custom organization is needed.
        """
        pass

    def _download_from_kaggle(
        self,
        dataset_id: Optional[str] = None,
        zip_filename: Optional[str] = None,
    ) -> None:
        """Download and extract dataset from Kaggle.

        Args:
            dataset_id: Kaggle dataset ID (uses _get_kaggle_dataset_id() if None)
            zip_filename: Name for downloaded zip file (auto-generated if None)
        """
        # Get dataset ID
        if dataset_id is None:
            dataset_id = self._get_kaggle_dataset_id()

        # Create directories
        self.root.mkdir(parents=True, exist_ok=True)

        # Generate zip filename if not provided
        if zip_filename is None:
            zip_filename = f"{dataset_id.split('/')[-1]}.zip"

        kaggle_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
        zip_path = self.root / zip_filename

        try:
            print(f"Downloading {dataset_id} from Kaggle...")
            print("Note: This requires Kaggle API authentication.")
            print(
                "Please ensure you have ~/.kaggle/kaggle.json with your API credentials."
            )

            # Download with progress bar
            self._download_with_progress(kaggle_url, zip_path)

            # Extract
            print("Extracting dataset...")
            self._extract_zip(zip_path)
            print("Extraction completed")

            # Clean up zip file
            zip_path.unlink()
            print("Cleaned up zip file")

            # Organize files (subclass-specific)
            self._organize_extracted_files()
            print("Dataset structure organized successfully")

        except requests.exceptions.RequestException as e:
            self._handle_download_error(e, dataset_id)
            raise

        except zipfile.BadZipFile as e:
            self._handle_zip_error(e, zip_path)
            raise

        except Exception as e:
            self._handle_generic_error(e, zip_path)
            raise

    def _download_with_progress(self, url: str, output_path: Path) -> None:
        """Download file from URL with progress display.

        Args:
            url: URL to download from
            output_path: Path to save downloaded file
        """
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
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                print()  # New line after progress
            else:
                # No content-length header, just download
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                print("Download completed (size unknown)")

        print(f"Download completed: {output_path}")

    def _extract_zip(self, zip_path: Path) -> None:
        """Extract zip file to dataset root.

        Args:
            zip_path: Path to zip file
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

    def _handle_download_error(self, error: Exception, dataset_id: str) -> None:
        """Handle download errors and show manual instructions.

        Args:
            error: The exception that occurred
            dataset_id: Kaggle dataset ID
        """
        print(f"\nError downloading dataset: {error}")
        print("Please check your internet connection and Kaggle API credentials.")
        print("\nManual download instructions:")

        instructions = self._get_manual_download_instructions()
        for instruction in instructions:
            print(instruction)

    def _handle_zip_error(self, error: Exception, zip_path: Path) -> None:
        """Handle zip extraction errors.

        Args:
            error: The exception that occurred
            zip_path: Path to the problematic zip file
        """
        print(f"\nError extracting zip file: {error}")
        if zip_path.exists():
            zip_path.unlink()
            print("Removed corrupted zip file")

    def _handle_generic_error(self, error: Exception, zip_path: Path) -> None:
        """Handle generic errors during download.

        Args:
            error: The exception that occurred
            zip_path: Path to zip file (may not exist)
        """
        print(f"\nUnexpected error during download: {error}")
        if zip_path.exists():
            zip_path.unlink()
            print("Cleaned up partial download")

    def _find_and_move_file(
        self, filename: str, search_patterns: Optional[list[str]] = None
    ) -> bool:
        """Find a file in dataset root (recursively) and move to root if needed.

        Args:
            filename: Name of file to find
            search_patterns: Alternative name patterns to search for

        Returns:
            True if file was found, False otherwise
        """
        expected_path = self.root / filename

        # Already in the right place
        if expected_path.exists():
            return True

        # Search for the file
        search_names = [filename]
        if search_patterns:
            search_names.extend(search_patterns)

        for search_name in search_names:
            for found_file in self.root.rglob(search_name):
                if found_file.is_file():
                    print(
                        f"Found {filename} at {found_file}, moving to {expected_path}"
                    )
                    shutil.move(str(found_file), str(expected_path))
                    return True

        return False

    def _find_and_move_directory(
        self, dirname: str, search_patterns: Optional[list[str]] = None
    ) -> bool:
        """Find a directory in dataset root (recursively) and move to root if needed.

        Args:
            dirname: Name of directory to find
            search_patterns: Alternative name patterns to search for

        Returns:
            True if directory was found, False otherwise
        """
        expected_path = self.root / dirname

        # Already in the right place
        if expected_path.exists() and expected_path.is_dir():
            return True

        # Search for the directory
        search_names = [dirname]
        if search_patterns:
            search_names.extend(search_patterns)

        for search_name in search_names:
            for found_dir in self.root.rglob(search_name):
                if found_dir.is_dir():
                    print(f"Found {dirname} at {found_dir}, moving to {expected_path}")
                    shutil.move(str(found_dir), str(expected_path))
                    return True

        return False
