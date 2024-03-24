from __future__ import annotations

from typing import TYPE_CHECKING, cast

import napari.layers
import napari.viewer
import numpy as np
import torch
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from rich import print
from segment_anything import SamPredictor, sam_model_registry
from skimage import measure

if TYPE_CHECKING:
    from segment_anything.modeling import Sam

mc = "/Users/fdrgsp/Documents/git/sam_vit_h_4b8939.pth"
mtype = "vit_h"

FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
EXTENDED = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class SamPromptWidget(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        viewer: napari.viewer.Viewer,
        model_checkpoint: str = "",
        model_type: str = "",
    ) -> None:
        super().__init__(parent)

        self._viewer = viewer

        self._sam: Sam | None = None
        self._predictor: SamPredictor | None = None

        # Add the model groupbox
        _model_group = QGroupBox("SAM Model Checkpoint")
        _model_group_layout = QGridLayout(_model_group)
        _model_group_layout.setSpacing(10)
        _model_group_layout.setContentsMargins(10, 10, 10, 10)

        _model_lbl = QLabel("Model Path:")
        _model_lbl.setSizePolicy(FIXED)
        self._model_le = QLineEdit(text=model_checkpoint)

        _model_type_lbl = QLabel("Model Type:")
        _model_type_lbl.setSizePolicy(FIXED)
        self._model_type_le = QLineEdit(text=model_type)

        self._model_browse_btn = QPushButton("Browse")
        self._model_browse_btn.setSizePolicy(FIXED)
        self._model_browse_btn.clicked.connect(self._browse_model)
        self._model_status_label = QLabel("Model not loaded.")
        self._load_modle_btn = QPushButton("Load")
        self._load_modle_btn.clicked.connect(self._on_load)

        _model_group_layout.addWidget(_model_lbl, 0, 0)
        _model_group_layout.addWidget(self._model_le, 0, 1)
        _model_group_layout.addWidget(self._model_browse_btn, 0, 2)
        _model_group_layout.addWidget(_model_type_lbl, 1, 0)
        _model_group_layout.addWidget(self._model_type_le, 1, 1, 1, 2)
        _model_group_layout.addWidget(self._model_status_label, 2, 0, 1, 2)
        _model_group_layout.addWidget(self._load_modle_btn, 2, 2)

        # add image groupbox
        _image_group = QGroupBox("Layer Selector")
        _image_group_layout = QGridLayout(_image_group)
        _image_combo_lbl = QLabel("Layer:")
        _image_combo_lbl.setSizePolicy(FIXED)
        self._image_combo = QComboBox()
        self._add_points_layer_btn = QPushButton("Add Point Layers")
        self._add_points_layer_btn.setSizePolicy(FIXED)
        self._add_points_layer_btn.clicked.connect(self._add_points_layers)
        _image_group_layout.addWidget(_image_combo_lbl, 0, 0)
        _image_group_layout.addWidget(self._image_combo, 0, 1)
        _image_group_layout.addWidget(self._add_points_layer_btn, 0, 2)

        # add mask predictor
        _predictor_group = QGroupBox("Predictor")
        _predictor_group_layout = QGridLayout(_predictor_group)
        self._standard_radio = QRadioButton("Standard Predictor")
        self._standard_radio.setChecked(True)
        self._loop_radio = QRadioButton("Loop Single Points Predictor")
        self._generate_mask_btn = QPushButton("Predict")
        self._generate_mask_btn.setSizePolicy(FIXED)
        self._generate_mask_btn.clicked.connect(self._on_predict)
        _predictor_group_layout.addWidget(self._standard_radio, 0, 0)
        _predictor_group_layout.addWidget(self._loop_radio, 0, 1)
        _predictor_group_layout.addWidget(self._generate_mask_btn, 0, 2)

        # add the widget to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(_model_group)
        main_layout.addWidget(_image_group)
        main_layout.addWidget(_predictor_group)

        # connections
        self._viewer.layers.events.changed.connect(self._on_layers_changed)
        self._viewer.layers.events.inserted.connect(self._on_layers_changed)
        self._viewer.layers.events.removed.connect(self._on_layers_changed)

    def _browse_model(self) -> None:
        """Open a file dialog to select the SAM Model Checkpoint."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select the SAM Model Checkpoint to use.", "", "pth(*.pth)"
        )
        if filename:
            self._model_le.setText(filename)

    def _on_load(self) -> None:
        """Load the SAM model."""
        self._sam = None
        self._predictor = None

        model_checkpoint = self._model_le.text()
        model_type = self._model_type_le.text()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
        except Exception as e:
            self._model_status_label.setText("Error while loading model!")
            self._sam = None
            self._predictor = None
            print(e)

        self._sam.to(device=device)

        self._model_status_label.setText(
            f"Model successfully loaded!  Using device: {device.upper()}."
        )

        self._predictor = SamPredictor(self._sam)

    def _on_layers_changed(self) -> None:
        """Update the layer combo box."""
        current_layer = self._image_combo.currentText()
        self._image_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._image_combo.addItem(layer.name)
        if current_layer and current_layer in self._viewer.layers:
            self._image_combo.setCurrentText(current_layer)

    def _add_points_layers(self) -> None:
        """Add the points layers to the viewer."""
        layer = self._image_combo.currentText()

        if not layer:
            return

        layers_meta = [
            (lay.metadata.get("id"), lay.metadata.get("type"))
            for lay in self._viewer.layers
        ]

        if (layer, 0) not in layers_meta:
            self._viewer.add_points(
                name=f"{layer}_points [BACKGROUND]",
                ndim=2,
                metadata={"id": layer, "type": 0},
            )

        if (layer, 1) not in layers_meta:
            self._viewer.add_points(
                name=f"{layer}_points [FOREGROUND]",
                ndim=2,
                metadata={"id": layer, "type": 1},
            )

    def _on_predict(self) -> None:
        """Start the prediction."""
        if self._predictor is None:
            return

        layer_name = self._image_combo.currentText()

        if (
            not self._viewer.layers
            or not layer_name
            or layer_name not in self._viewer.layers
        ):
            return

        frg_point_layer, bkg_point_layer = self._get_point_layers(layer_name)

        if frg_point_layer is None or bkg_point_layer is None:
            return

        frg_points: list[tuple[tuple[int, int], int]] = []
        for p in frg_point_layer.data:
            x, y = p[1], p[0]
            frg_points.append(((x, y), 1))

        bkg_points: list[tuple[tuple[int, int], int]] = []
        for p in bkg_point_layer.data:
            x, y = int(p[1]), int(p[0])
            bkg_points.append(((x, y), 0))

        if not frg_points and not bkg_points:
            return

        self._predict(layer_name, frg_points, bkg_points)

    def _get_point_layers(
        self, layer_name: str
    ) -> tuple[napari.layers.Points | None, napari.layers.Points, None]:
        """Get the layer from the viewer."""
        frg_point_layer = None
        bkg_point_layer = None

        for layer in self._viewer.layers:
            if (
                isinstance(layer, napari.layers.Points)
                and layer.metadata.get("id") == layer_name
            ):
                if layer.metadata.get("type") == 1:
                    frg_point_layer = layer
                elif layer.metadata.get("type") == 0:
                    bkg_point_layer = layer

        return frg_point_layer, bkg_point_layer

    def _predict(
        self,
        layer_name: str,
        foreground_points: list[tuple[tuple[int, int], int]],
        background_points: list[tuple[tuple[int, int], int]],
    ) -> None:
        """Run the SamPredictor."""
        image = self._convert_image_to_8bit(layer_name)
        self._predictor.set_image(image)

        if self._standard_radio.isChecked():
            masks, scores = self._standard_predictor(
                foreground_points, background_points
            )
        else:
            masks, scores = self._loop_predictor(foreground_points)

        if console := getattr(self._viewer.window._qt_viewer, "console", None):
            console.push({"masks": masks, "scores": scores})

        self._display_labels(layer_name, masks)

    def _standard_predictor(
        self,
        foreground_points: list[tuple[tuple[int, int], int]],
        background_points: list[tuple[tuple[int, int], int]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """The Standard SAM Predictor.

        Feed foreground and background points to the predictor in a list and get the
        masks and scores.
        """
        input_point = []
        input_label = []
        for point, label in foreground_points:
            input_point.append(point)
            input_label.append(label)
        for bg_points, bg_label in background_points:
            input_point.append(bg_points)
            input_label.append(bg_label)

        input_point = np.array(input_point)
        input_label = np.array(input_label)

        masks, score, _ = self._predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return masks, score

    def _loop_predictor(
        self, foreground_points: list[tuple[tuple[int, int], int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """The Loop SAM Predictor.

        Feed each foreground point to the predictor individually and get the masks and
        scores for each point.
        """
        masks: list[np.ndarray] = []
        scores: list[float] = []
        for point, label in foreground_points:
            input_point = np.array([point])
            input_label = np.array([label])

            mask, score, _ = self._predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            masks.append(mask)
            scores.append(score)
        return masks, scores

    def _convert_image_to_8bit(self, layer_name: str) -> np.ndarray:
        """Convert the image to 8-bit and stack to 3 channels."""
        # TODO: Handle already 8-bit, rgb images + stacks
        layer = cast(napari.layers.Image, self._viewer.layers[layer_name])
        data = layer.data
        # Normalize to the range 0-1
        img_normalized = data / np.max(data)
        # Scale to 8-bit (0-255)
        img_8bit = (img_normalized * 255).astype(np.uint8)
        # Stack the image three times to create a 3-channel image
        img_8bit = np.stack((img_8bit, img_8bit, img_8bit), axis=-1)
        return img_8bit

    def _display_labels(self, layer_name: str, masks: list[np.ndarray]) -> None:
        """Display the masks as labels in the viewer."""
        if len(masks) == 1:
            labeled_mask = measure.label(masks[0])
            self._viewer.add_labels(labeled_mask, name=f"{layer_name}_mask")
            return

        final_mask = np.zeros_like(masks[0], dtype=np.int32)
        for mask in masks:
            labeled_mask = measure.label(mask)
            labeled_mask[labeled_mask != 0] += final_mask.max()
            final_mask += labeled_mask
        self._viewer.add_labels(final_mask, name=f"{layer_name}_mask")
