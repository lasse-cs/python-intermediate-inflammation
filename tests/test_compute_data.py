from unittest.mock import Mock
import numpy as np


def test_analyse_data_mock_source():
    from inflammation.compute_data import analyse_data
    data_source = Mock()

    data_source.load_inflammation_data.return_value = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    analyse_data(data_source)
