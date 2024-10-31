import json
from pathlib import Path
from typing import Dict

import pandas as pd

from ..core.exceptions import AutofillErrorCode, AutofillException
from ..core.interfaces import DataReader
from ..utils.data_utils import convert_numeric_to_int64


class FileDataReader(DataReader):
    def read_csv(self, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return convert_numeric_to_int64(df)
        except Exception as e:
            raise AutofillException(
                AutofillErrorCode.FILE_ERROR,
                f"Failed to read CSV file: {str(e)}",
                source_file=file_path,
            )

    def read_json(self, file_path: Path) -> Dict:
        try:
            with file_path.open("r") as f:
                return json.load(f)
        except Exception as e:
            raise AutofillException(
                AutofillErrorCode.FILE_ERROR,
                f"Failed to read JSON file: {str(e)}",
                source_file=file_path,
            )
