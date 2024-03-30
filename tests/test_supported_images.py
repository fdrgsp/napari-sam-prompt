from unittest.mock import patch

import napari
import numpy as np
import pytest
from napari_sam_prompt._sam_prompt_widget import SamPromptWidget


@pytest.fixture
def widget():
    widget = SamPromptWidget(viewer=napari.Viewer())
    with patch.object(widget, "_console", return_value=None):
        yield widget


data = [
    # imagenp.ndarray,              x_axis_index,   y_axis_index
    (np.random.rand(100, 100), 0, 1),
    # (np.random.rand(100, 100, 3),   0,              1),
    # (np.random.rand(100, 100, 6),   0,              1),
    # (np.random.rand(2, 100, 100),   0,              1),
]


@pytest.mark.parametrize("data", data)
def test_supported_images(widget: SamPromptWidget, data: tuple[np.ndarray, int, int]):
    image, idx_x, idx_y = data

    viewer = widget._viewer
    viewer.add_image(image, name="test_layer")

    # Call the method to test
    result = widget._convert_image("test_layer")

    # Check the result
    # The result should have an extra dimension for the 3 channels
    assert result.shape == (image.shape[idx_x], image.shape[idx_y], 3)
    # The result should be 8-bit
    assert result.dtype == np.uint8
    # # The result should be in the range [0, 255]
    assert np.all(result >= 0) and np.all(result <= 255)
