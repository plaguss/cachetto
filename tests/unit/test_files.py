import pickle
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from cachetto._files import read_cached_file, save_to_file


class TestReadCachedFile:
    @pytest.mark.parametrize(
        "data",
        [
            {"foo": "bar"},
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            {"meta": "info", "df": pd.DataFrame({"x": [10, 20]})},
        ],
    )
    def test_read_cached_file_success(self, tmp_path, data):
        filename = tmp_path / "testfile.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f)

        result = read_cached_file(filename)

        if isinstance(data, pd.DataFrame):
            pd.testing.assert_frame_equal(result, data)
        else:
            if any(isinstance(v, pd.DataFrame) for v in data.values()):
                pd.testing.assert_frame_equal(result["df"], data["df"])
            else:
                assert result == data

    @pytest.mark.parametrize(
        "exception",
        [
            pickle.UnpicklingError("bad pickle"),
            EOFError("unexpected EOF"),
            FileNotFoundError("no such file"),
            PermissionError("no permission"),
            AttributeError("bad attribute"),
            ModuleNotFoundError("module not found"),
            OSError("os error"),
        ],
    )
    def test_read_cached_file_failure(self, exception):
        mock_path = MagicMock(spec=Path)
        mock_path.unlink = MagicMock()

        with patch("builtins.open", mock_open()) as _:
            with patch("pickle.load", side_effect=exception):
                result = read_cached_file(mock_path)

        assert result is None
        mock_path.unlink.assert_called_once_with(missing_ok=True)


class TestSaveToFile:
    def test_save_to_file_success(self, tmp_path):
        result = {"key": "value"}
        filename = tmp_path / "output.pickle"

        save_to_file(result, filename)

        assert filename.exists()
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == result

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump", side_effect=pickle.PicklingError("Pickling failed"))
    def test_save_to_file_pickling_error(self, mock_pickle, mock_open):
        mock_path = MagicMock(spec=Path)
        mock_path.unlink = MagicMock()

        save_to_file(object(), mock_path)

        mock_path.unlink.assert_called_once_with(missing_ok=True)
