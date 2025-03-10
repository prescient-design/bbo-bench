from unittest.mock import MagicMock, patch

from bbo_bench.utils import load_presolved_data


class TestLoadPresolvedData:
    def setup_method(self):
        # Create mock black box
        self.black_box = MagicMock()
        self.black_box.alphabet = ["A", "B", "C", "D"]

        # Test parameters
        self.url = "https://example.com/data.tar.gz"
        self.s3_bucket = "s3://my-bucket"
        self.folder_name = "data"

    @patch("bbo_bench.utils._load_presolved_data._load_url_data")
    def test_load_presolved_data_from_url(self, mock_load_url_data):
        # Setup mock return values
        ehrlich_data = [{"num_states": 8, "dim": 128, "num_motifs": 8}]
        presolved_data = [
            {"particle": "[0, 1, 2, 3]", "score": "-0.5"},
            {"particle": "[3, 2, 1, 0]", "score": "-0.7"},
        ]
        mock_load_url_data.return_value = (ehrlich_data, presolved_data)

        # Call function
        x, y = load_presolved_data(self.url, self.folder_name, self.black_box)

        # Check that _load_url_data was called with correct args
        mock_load_url_data.assert_called_once_with(self.url, self.folder_name)

        # Check return values
        assert x.shape == (2, 4)  # 2 sequences, each with 4 elements
        assert y.shape == (2, 1)  # 2 scores
        assert x[0, 0] == "A"  # First element of first sequence
        assert x[1, 0] == "D"  # First element of second sequence
        assert y[0, 0] == 0.5  # First score (negated)
        assert y[1, 0] == 0.7  # Second score (negated)

    @patch("bbo_bench.utils._load_presolved_data._load_s3_data")
    def test_load_presolved_data_from_s3(self, mock_load_s3_data):
        # Setup mock return values
        ehrlich_data = [{"num_states": 8, "dim": 128, "num_motifs": 8}]
        presolved_data = [
            {"particle": "[0, 1, 2, 3]", "score": "-0.5"},
            {"particle": "[3, 2, 1, 0]", "score": "-0.7"},
        ]
        mock_load_s3_data.return_value = (ehrlich_data, presolved_data)

        # Call function
        x, y = load_presolved_data(
            self.s3_bucket, self.folder_name, self.black_box
        )

        # Check that _load_s3_data was called with correct args
        mock_load_s3_data.assert_called_once_with(
            self.s3_bucket, self.folder_name
        )

        # Check return values (same as above)
        assert x.shape == (2, 4)
        assert y.shape == (2, 1)
