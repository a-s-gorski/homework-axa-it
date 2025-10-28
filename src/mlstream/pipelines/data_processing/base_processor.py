from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseProcessor(ABC):
    """
    Abstract base class for data processors in the MLStream pipeline.

    Subclasses should implement `process` and may optionally implement `_validate`
    for runtime data validation.
    """

    def _validate(self) -> bool:
        """
        Validate the input data (override in subclasses).

        Returns:
            bool: True if validation passes. Should raise on failure in practice.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process the input data and return the processed result.

        Args:
            data: The input to be processed (e.g., a pandas DataFrame).

        Returns:
            Any: The processed data.
        """
        ...
