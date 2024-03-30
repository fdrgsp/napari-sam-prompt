from __future__ import annotations

import cv2 as cv
import numpy as np


def _convert_8bit(data: np.ndarray) -> np.ndarray:
    """Convert image to 8-bit."""
    data = (255 * ((data - data.min()) / (data.max() - data.min()))).astype("uint8")
    return data


def _convert_to_three_channels(data: np.ndarray) -> np.ndarray:
    """Convert the image to 3 channels."""
    if len(data.shape) == 3:
        # Convert the image from CHW to HWC dimension
        ch_loc = np.argmin(data.shape)
        if ch_loc == 0:
            data = np.transpose(data, (1, 2, 0))
        channels = data.shape[-1]
    else:
        channels = 1

    data_8bit = np.zeros_like(data, dtype="uint8")

    if channels == 1:
        data_8bit = _convert_8bit(data)
        data_8bit = cv.equalizeHist(data_8bit)
        return np.stack([data_8bit] * 3, axis=-1)
    elif channels == 2:
        for i in range(channels):
            temp_data = data[:, :, i]
            temp_data = _convert_8bit(temp_data)
            data_8bit[:, :, i] = cv.equalizeHist(temp_data)
        avg_channels = np.mean(data_8bit, axis=-1, dtype="uint8")
        return np.stack([data_8bit[:, :, 0], data_8bit[:, :, 1], avg_channels], axis=-1)
    elif channels == 3:
        for i in range(channels):
            temp_data = data[:, :, i]
            temp_data = _convert_8bit(temp_data)
            data_8bit[:, :, i] = cv.equalizeHist(temp_data)
        return data_8bit
    else:
        for i in range(channels):
            temp_data = data[:, :, i]
            temp_data = _convert_8bit(temp_data)
            data_8bit[:, :, i] = cv.equalizeHist(temp_data)
        avg_channels = np.mean(data_8bit, axis=-1, dtype="uint8")
        return np.stack([avg_channels] * 3, axis=-1)
