import numpy as np
import pytest
from napari_sam_prompt._util import _convert_8bit, _convert_to_three_channels

data = [
    # imagenp.ndarray, x_axis_index, y_axis_index
    (np.random.rand(100, 100), 0, 1),
    (np.random.rand(100, 100, 3), 0, 1),
    (np.random.rand(100, 100, 2), 0, 1),
    (np.random.rand(100, 100, 6), 0, 1),
    # (np.random.rand(2, 100, 100), 0, 1),
]


@pytest.mark.parametrize("data", data)
def test_supported_images(data: tuple[np.ndarray, int, int]):
    image, idx_x, idx_y = data

    # Call the method to test
    three_ch = _convert_to_three_channels(image)
    result = _convert_8bit(three_ch)

    # Check the result
    # The result should have an extra dimension for the 3 channels
    assert result.shape == (image.shape[idx_x], image.shape[idx_y], 3)
    # The result should be 8-bit
    assert result.dtype == np.uint8
    # # The result should be in the range [0, 255]
    assert np.all(result >= 0) and np.all(result <= 255)
