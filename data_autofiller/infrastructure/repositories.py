from pathlib import Path
from typing import Dict, Optional

from ..core.exceptions import AutofillErrorCode, AutofillException
from ..core.interfaces import QuestionRepository
from ..utils.file_utils import read_json_file


class FileQuestionRepository(QuestionRepository):
    def __init__(self, questions_dir: Path):
        self.questions_dir = questions_dir
        self._cache: Dict = {}

    def _read_json(self, file_path: Path) -> Dict:
        return read_json_file(file_path)

    def load_questions(self, directory: Path) -> Dict:
        if self._cache:
            return self._cache

        try:
            for json_file in directory.glob("*.json"):
                questions = self._read_json(json_file)
                self._cache.update(questions)
            return self._cache
        except Exception as e:
            raise AutofillException(
                AutofillErrorCode.FILE_ERROR,
                f"Failed to load questions: {str(e)}",
                source_file=directory,
            )

    def get_question(self, question_id: str) -> Optional[Dict]:
        if not self._cache:
            self.load_questions(self.questions_dir)
        return self._cache.get(question_id)

    def clear_cache(self):
        self._cache.clear()

    def __del__(self):
        self.clear_cache()
