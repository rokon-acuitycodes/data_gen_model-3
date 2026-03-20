from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class DataGenerator(ABC):
    """Abstract base class for data generators."""

    @abstractmethod
    def generate(self, original_file: Any, num_files: int = 100, **kwargs) -> List[Tuple[str, bytes]]:
        """
        Generate synthetic data based on the original file.

        Args:
            original_file: The original file object or data.
            num_files: Number of synthetic files to generate.
            **kwargs: Additional parameters.

        Returns:
            List of tuples: (filename, file_data as bytes)
        """
        pass
