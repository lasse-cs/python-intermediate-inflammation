"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import (
    daily_mean,
    daily_max,
    daily_min,
    patient_normalise,
)


@pytest.mark.parametrize(
    "test, expected",
    (([[0, 0], [0, 0], [0, 0]], [0, 0]), ([[1, 2], [3, 4], [5, 6]], [3, 4])),
)
def test_daily_mean(test, expected):
    """
    Test that mean function works for an array of zeros and positive integers.
    """

    test_input = np.array(test)
    test_result = np.array(expected)

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    (([[0, 0], [0, 0], [0, 0]], [0, 0]), ([[1, 2], [3, 4], [5, 6]], [5, 6])),
)
def test_daily_max(test, expected):
    """
    Test that max function works for an array of zeros and positive integers.
    """

    test_input = np.array(test)
    test_result = np.array(expected)
    npt.assert_array_equal(daily_max(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    (([[0, 0], [0, 0], [0, 0]], [0, 0]), ([[1, 2], [3, 4], [5, 6]], [1, 2])),
)
def test_daily_min(test, expected):
    """
    Test that min function works for an array of zeros and positive integers.
    """

    test_input = np.array(test)
    test_result = np.array(expected)
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_strings():
    """Test for TypeError when passing strings."""

    with pytest.raises(TypeError):
        daily_min(np.array([["a", "b"], ["c", "d"], ["e", "f"]]))


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    (
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[-1, -2, 2], [0, 0, float("nan")]],
            [[0, 0, 1], [0, 0, 0]],
            ValueError,
        ),
    ),
)
def test_patient_normalise(test, expected, expect_raises):
    """
    Test that normalisation works for arrays of one and positive integers.
    Test with a relative and absolute tolerance of 0.01.
    """
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            result = patient_normalise(np.array(test))
            npt.assert_allclose(
                result,
                np.array(expected),
                rtol=0.01,
                atol=0.01,
            )
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=0.01, atol=0.01)
