import numpy as np
import pytest
from napari.layers import Image
from napari_sam_prompt._util import _convert_image

data = [
    (Image(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), rgb=True)),  # RGB
    (Image(np.random.rand(100, 100), rgb=False)),  # 2D image
    (Image(np.random.rand(1, 100, 100), rgb=False)),  # 2D image with 1 channel idx 0
    (Image(np.random.rand(3, 100, 100), rgb=False)),  # 2D image with 3 channel idx 0
    (Image(np.random.rand(100, 100, 1), rgb=False)),  # 2D image with 1 channel idx 2
    (Image(np.random.rand(100, 100, 2), rgb=False)),  # 2D image with 2 channels idx 2
    (Image(np.random.rand(100, 100, 6), rgb=False)),  # 3D image with 6 channels
]


@pytest.mark.parametrize("image", data)
def test_supported_images(image: Image):
    # Call the method to test
    result = _convert_image(image)

    # Check the result
    # The result should have an extra dimension for the 3 channels at index 2
    assert np.argmin(result.shape) == 2
    # The result should be 8-bit
    assert result.dtype == np.uint8
    # # The result should be in the range [0, 255]
    assert np.all(result >= 0) and np.all(result <= 255)
