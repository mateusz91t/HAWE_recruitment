import numpy as np
import pandas as pd
import pytest
from functions_and_data_cleaning_dir.additional_functions import moving_mean


@pytest.mark.parametrize(
    "source_array, window, result_array",
    [
        [
            [1, 2, 1, 2, 1, 2, 10, 20, 10, 20, 10, 20],
            4,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    1.5,
                    1.5,
                    1.5,
                    3.75,
                    8.25,
                    10.5,
                    15.0,
                    15.0,
                    15.0,
                ]
            ),
        ],
        [
            np.array([*range(1, 11)] + [*range(11, 0, -1)]),
            5,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    9.6,
                    9.8,
                    9.6,
                    9.0,
                    8.0,
                    7.0,
                    6.0,
                    5.0,
                    4.0,
                    3.0,
                ]
            ),
        ],
    ],
)
def test_moving_mean(source_array, window, result_array):
    """
    Checks moving_mean base on arrays.

    Tests of `functions_and_data_cleaning_dir.additional_functions.moving_mean`.
    Test checks if the values from sample and result are the same or are NaN.
    """
    a1 = np.array(
        moving_mean(pd.DataFrame(source_array), window)[
            f"0__{window}__mean"
        ].values
    )
    # are they the same         or are they a NaN together      in a whole array
    assert ((a1 == result_array) | np.isnan(a1) & np.isnan(result_array)).all()
