from unittest.mock import patch

import pandas as pd

from cachetto._files import read_cached_file, save_to_file


class TestReadCachedFile:
    def testread_cached_file_success(self, tmp_path) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        file_path = tmp_path / "test.parquet"
        df.to_parquet(file_path)
        result = read_cached_file(file_path)
        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def testread_cached_file_failure(self, tmp_path) -> None:
        file_path = tmp_path / "corrupt.parquet"
        # Create a dummy file that will raise an error when read
        file_path.write_text("this is not a parquet file")

        with patch(
            "pandas.read_parquet", side_effect=Exception("Read failed")
        ) as mock_read:
            read_cached_file(file_path)
            mock_read.assert_called_once()
            assert not file_path.exists()


class TestSaveToFile:
    def testsave_to_file_success(self, tmp_path) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        file_path = tmp_path / "output.parquet"
        save_to_file(df, file_path)

        assert file_path.exists()
        loaded = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(loaded, df)

    def testsave_to_file_failure(self, tmp_path) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        file_path = tmp_path / "fail.parquet"

        with patch.object(
            df, "to_parquet", side_effect=Exception("Write failed")
        ) as mock_write:
            save_to_file(df, file_path)
            mock_write.assert_called_once()
            assert not file_path.exists()
