import json
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..core.exceptions import AutofillErrorCode, AutofillException
from ..core.interfaces import DataReader


class FileDataReader(DataReader):
    def read_csv(
        self, file_path: Path, chunksize: Optional[int] = None
    ) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        try:
            return pd.read_csv(
                file_path, chunksize=chunksize, dtype_backend="numpy_nullable"
            )
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
