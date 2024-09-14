import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import os
from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader()

    @patch('os.makedirs')
    def test_create_statistics_folder(self, mock_makedirs):
        folder_path = self.data_loader.create_statistics_folder()
        mock_makedirs.assert_called_once_with(folder_path, exist_ok=True)
        self.assertIsInstance(folder_path, str)

    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv):
        mock_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_read_csv.return_value = mock_df

        result = self.data_loader.load_data('fake_path.csv')

        mock_read_csv.assert_called_once_with('fake_path.csv')
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('os.path.exists')
    def test_load_data_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_data('non_existent_file.csv')

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_load_data_empty_file(self, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame()

        with self.assertRaises(ValueError):
            self.data_loader.load_data('empty_file.csv')

    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_load_data_parser_error(self, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        mock_read_csv.side_effect = pd.errors.ParserError("Parser error")

        with self.assertRaises(ValueError):
            self.data_loader.load_data('invalid_file.csv')

if __name__ == '__main__':
    unittest.main()