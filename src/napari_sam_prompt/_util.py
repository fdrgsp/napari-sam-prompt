from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from napari.layers import Image


def _convert_image(layer: Image) -> np.ndarray:
    """Convert the image to 3 channels."""
    # if rgb image, return the data
    if layer.rgb:
        return layer.data

    # if data is 2D, convert to shape (x, y, 3)
    if layer.ndim == 2:
        # Stack the image three times to create a 3-channel image by stacking the image
        # three times
        data_8bit = _convert_8bit(layer.data)
        return np.stack(([data_8bit] * 3), axis=-1)

    # if data is 3D
    if layer.ndim == 3:
        return _handle_3d_layers(layer)

    if layer.ndim > 3:
        raise ValueError("Only 2D and 3D images are supported.")


def _handle_3d_layers(layer: Image) -> np.ndarray:
    """Convert 3D image to 3 channels usable for SAM."""
    data = layer.data

    # Convert the image from Ch dimension from position 0 to position -1
    ch_loc = np.argmin(data.shape)
    if ch_loc == 0:
        data = np.transpose(data, (1, 2, 0))

    # Get the number of channels
    channels = data.shape[-1]

    # if shape is (x, y, 1), convert to shape (x, y, 3) by stacking the image three
    # times and convert the image to 8-bit
    if channels == 1:
        data_8bit = _convert_8bit(data)
        return np.stack([data_8bit] * 3, axis=-1)

    # if shape is (x, y, 2), average the two channels to crteate a 3rd channel and
    # convert the image to 8-bit
    if channels == 2:
        data_8bit = np.zeros_like(data, dtype="uint8")
        for i in range(channels):
            temp_data = data[:, :, i]
            temp_data = _convert_8bit(temp_data)
            data_8bit[:, :, i] = temp_data
        avg_channels = np.mean(data_8bit, axis=-1, dtype="uint8")
        return np.stack([data_8bit[:, :, 0], data_8bit[:, :, 1], avg_channels], axis=-1)

    # if shape is (x, y, 3), return the data converted to 8-bit
    if channels == 3:
        data_8bit = np.zeros_like(data, dtype="uint8")
        for i in range(channels):
            temp_data = data[:, :, i]
            temp_data = _convert_8bit(temp_data)
            data_8bit[:, :, i] = temp_data
        return data_8bit

    # if shape is (x, y, >3), average the channels to create a 3 channel image and
    # convert the image to 8-bit
    data_8bit = np.zeros_like(data, dtype="uint8")
    for i in range(channels):
        temp_data = data[:, :, i]
        temp_data = _convert_8bit(temp_data)
        data_8bit[:, :, i] = temp_data
    avg_channels = np.mean(data_8bit, axis=-1, dtype="uint8")
    return np.stack([avg_channels] * 3, axis=-1)


def _convert_8bit(data: np.ndarray) -> np.ndarray:
    """Convert image to 8-bit."""
    data = (255 * ((data - data.min()) / (data.max() - data.min()))).astype("uint8")
    return data
