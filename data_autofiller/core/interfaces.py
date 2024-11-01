from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class DataReader(ABC):
    @abstractmethod
    def read_csv(
        self, file_path: Path, chunksize: Optional[int] = None
    ) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        pass

    @abstractmethod
    def read_json(self, file_path: Path) -> Dict:
        pass


class QuestionRepository(ABC):
    @abstractmethod
    def load_questions(self, directory: Path) -> Dict:
        pass

    @abstractmethod
    def get_question(self, question_id: str) -> Optional[Dict]:
        pass


class RuleEngine(ABC):
    @abstractmethod
    def process_rules(
        self, question_id: str, answer: str, questions_data: Dict
    ) -> Dict[str, str]:
        pass

    @abstractmethod
    def process_chunk(self, chunk: pd.DataFrame, questions_data: Dict) -> pd.DataFrame:
        pass
