import logging
from pathlib import Path

import pandas as pd
import rdata
from kedro.io.core import AbstractDataset

logger = logging.getLogger(__name__)


class RdaDataset(AbstractDataset):
    def __init__(
        self,
        filepath: str,
        object_name: str,
        output_directory: str = "",
        version=None,
    ):
        """
        Args:
            filepath: Path to the local .rda file.
            object_name: The name of the object inside the .rda (e.g. 'pg15training').
            output_directory: Optional directory to save the CSV version.
            version: Optional Kedro dataset version.
        """
        self._filepath: Path = Path(filepath)
        self._object_name = object_name
        self._output_directory: Path = (
            Path(output_directory) if output_directory else Path("")
        )
        self._version = version

    def load(self) -> pd.DataFrame:
        """Load the R object from an existing .rda file and return it as a DataFrame."""
        if not self._exists():
            raise FileNotFoundError(f"Dataset not found at {self._filepath}")

        logger.info(f"Loading RDA file from: {self._filepath}")

        with open(self._filepath, "rb") as f:
            r_data = rdata.read_rda(f)[self._object_name]

        if not isinstance(r_data, pd.DataFrame):
            raise TypeError(f"Object '{self._object_name}' is not a DataFrame.")

        if self._output_directory:
            csv_path = self._output_directory / f"{self._object_name}.csv"
            self._save(r_data, csv_path)

        return r_data

    def _save(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Save DataFrame to CSV at the given filepath."""
        df.to_csv(str(output_path), index=False)
        logger.info(f"Saved dataset to {output_path}")

    def _exists(self) -> bool:
        """Check if the .rda file exists locally."""
        exists = self._filepath.exists()
        return exists

    def _describe(self):
        return dict(
            filepath=str(self._filepath),
            object_name=self._object_name,
            output_directory=(
                str(self._output_directory) if self._output_directory else None
            ),
        )
