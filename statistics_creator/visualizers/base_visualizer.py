from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
from logger import log_execution_time


class BaseVisualizer(ABC):
    """
    Abstract base class for data visualizers.
    """

    @abstractmethod
    @log_execution_time
    def visualize(self, data: Any, output_path: str) -> None:
        """
        Abstract method to visualize data.
        """
        pass

    def save_plot(self, output_path: str, dpi: int = 300) -> None:
        """
        Save the current matplotlib plot to a file.
        """
        plt.savefig(output_path, dpi=dpi)
        plt.close()
