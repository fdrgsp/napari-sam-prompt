from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, cast

import napari.layers
import napari.viewer
import numpy as np
import torch
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from rich import print
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from skimage import measure
from superqt.utils import create_worker, ensure_main_thread

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

        self._console = getattr(self._viewer.window._qt_viewer, "console", None)

        self._sam: Sam | None = None
        self._predictor: SamPredictor | None = None
        self._mask_generator: SamAutomaticMaskGenerator | None = None

        self._success = False

        # Add the model groupbox
        self._model_group = QGroupBox("SAM Model Checkpoint")
        _model_group_layout = QGridLayout(self._model_group)
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

        # add layer selector groupbox
        self._layer_group = QGroupBox("Layer Selector")
        _image_group_layout = QGridLayout(self._layer_group)
        _image_combo_lbl = QLabel("Layer:")
        _image_combo_lbl.setSizePolicy(FIXED)
        self._image_combo = QComboBox()
        _image_group_layout.addWidget(_image_combo_lbl, 0, 0)
        _image_group_layout.addWidget(self._image_combo, 0, 1)

        #  add automatic segmentation
        self._automatic_seg_group = QGroupBox("SAM Automatic Mask Generator")
        _automatic_group_layout = QGridLayout(self._automatic_seg_group)

        _points_per_side_lbl = QLabel("Points per side:")
        self._points_per_side = QSpinBox()
        self._points_per_side.setValue(32)

        _points_per_batch_lbl = QLabel("Points per batch:")
        self._points_per_batch = QSpinBox()
        self._points_per_batch.setValue(64)

        _pred_iou_thresh_lbl = QLabel("Pred IOU Threshold:")
        self._pred_iou_thresh = QDoubleSpinBox()
        self._pred_iou_thresh.setValue(0.88)

        _stability_score_thresh_lbl = QLabel("Stability Score Threshold:")
        self._stability_score_thresh = QDoubleSpinBox()
        self._stability_score_thresh.setValue(0.95)

        _stability_score_offset_lbl = QLabel("Stability Score Offset:")
        self._stability_score_offset = QDoubleSpinBox()
        self._stability_score_offset.setValue(1.0)

        _box_nms_thresh_lbl = QLabel("Box NMS Threshold:")
        self._box_nms_thresh = QDoubleSpinBox()
        self._box_nms_thresh.setValue(0.7)

        _crop_n_layers_lbl = QLabel("Crop N Layers:")
        self._crop_n_layers = QSpinBox()
        self._crop_n_layers.setValue(0)

        _crop_nms_thresh_lbl = QLabel("Crop NMS Threshold:")
        self._crop_nms_thresh = QDoubleSpinBox()
        self._crop_nms_thresh.setValue(0.7)

        _crop_overlap_ratio_lbl = QLabel("Crop Overlap Ratio:")
        self._crop_overlap_ratio = QDoubleSpinBox()
        self._crop_overlap_ratio.setValue(512 / 1500)

        _crop_n_points_downscale_factor_lbl = QLabel("Crop N Points Downscale Factor:")
        self._crop_n_points_downscale_factor = QSpinBox()
        self._crop_n_points_downscale_factor.setValue(1)

        _min_mask_region_area_lbl = QLabel("Min Mask Region Area:")
        self._min_mask_region_area = QSpinBox()
        self._min_mask_region_area.setValue(0)

        _output_mode_lbl = QLabel("Output Mode:")
        self._output_mode = QLineEdit(text="binary_mask")

        _min_area_lbl = QLabel("Minimum Area:")
        self._min_area = QSpinBox()
        self._min_area.setMaximum(1000000)
        self._min_area.setValue(100)

        _max_area_lbl = QLabel("Maximum Area:")
        self._max_area = QSpinBox()
        self._max_area.setMaximum(1000000)
        self._max_area.setValue(10000)

        self._generate_mask_btn = QPushButton("Generate Masks")
        self._generate_mask_btn.clicked.connect(self._on_generate)

        _automatic_group_layout.addWidget(_points_per_side_lbl, 0, 0)
        _automatic_group_layout.addWidget(self._points_per_side, 0, 1)
        _automatic_group_layout.addWidget(_points_per_batch_lbl, 1, 0)
        _automatic_group_layout.addWidget(self._points_per_batch, 1, 1)
        _automatic_group_layout.addWidget(_pred_iou_thresh_lbl, 2, 0)
        _automatic_group_layout.addWidget(self._pred_iou_thresh, 2, 1)
        _automatic_group_layout.addWidget(_stability_score_thresh_lbl, 3, 0)
        _automatic_group_layout.addWidget(self._stability_score_thresh, 3, 1)
        _automatic_group_layout.addWidget(_stability_score_offset_lbl, 4, 0)
        _automatic_group_layout.addWidget(self._stability_score_offset, 4, 1)
        _automatic_group_layout.addWidget(_box_nms_thresh_lbl, 5, 0)
        _automatic_group_layout.addWidget(self._box_nms_thresh, 5, 1)
        _automatic_group_layout.addWidget(_crop_n_layers_lbl, 6, 0)
        _automatic_group_layout.addWidget(self._crop_n_layers, 6, 1)
        _automatic_group_layout.addWidget(_crop_nms_thresh_lbl, 7, 0)
        _automatic_group_layout.addWidget(self._crop_nms_thresh, 7, 1)
        _automatic_group_layout.addWidget(_crop_overlap_ratio_lbl, 8, 0)
        _automatic_group_layout.addWidget(self._crop_overlap_ratio, 8, 1)
        _automatic_group_layout.addWidget(_crop_n_points_downscale_factor_lbl, 9, 0)
        _automatic_group_layout.addWidget(self._crop_n_points_downscale_factor, 9, 1)
        _automatic_group_layout.addWidget(_min_mask_region_area_lbl, 10, 0)
        _automatic_group_layout.addWidget(self._min_mask_region_area, 10, 1)
        _automatic_group_layout.addWidget(_output_mode_lbl, 11, 0)
        _automatic_group_layout.addWidget(self._output_mode, 11, 1)
        _automatic_group_layout.addWidget(_min_area_lbl, 12, 0)
        _automatic_group_layout.addWidget(self._min_area, 12, 1)
        _automatic_group_layout.addWidget(_max_area_lbl, 13, 0)
        _automatic_group_layout.addWidget(self._max_area, 13, 1)
        _automatic_group_layout.addWidget(self._generate_mask_btn, 14, 0, 1, 2)

        # add mask predictor
        self._predictor_group = QGroupBox("SAM Predictor")
        _predictor_group_layout = QGridLayout(self._predictor_group)
        self._standard_radio = QRadioButton("Standard Predictor")
        self._standard_radio.setChecked(True)
        self._standard_radio.setSizePolicy(FIXED)
        self._loop_radio = QRadioButton("Loop Single Points Predictor")
        self._loop_radio.setSizePolicy(FIXED)
        self._add_points_layer_btn = QPushButton("Add Point Layers")
        self._add_points_layer_btn.setSizePolicy(FIXED)
        self._add_points_layer_btn.clicked.connect(self._add_points_layers)
        self._predict_btn = QPushButton("Predict")
        self._predict_btn.clicked.connect(self._on_predict)
        _predictor_group_layout.addWidget(self._standard_radio, 0, 0)
        _predictor_group_layout.addWidget(self._loop_radio, 0, 1)
        _predictor_group_layout.addWidget(self._add_points_layer_btn, 0, 2)
        _predictor_group_layout.addWidget(self._predict_btn, 1, 0, 1, 3)

        # info group
        _info_group = QGroupBox("Info")
        _info_group_layout = QVBoxLayout(_info_group)
        self._load_info_lbl = QLabel("Model not loaded.")
        self._info_lbl = QLabel()
        _info_group_layout.addWidget(self._load_info_lbl)
        _info_group_layout.addWidget(self._info_lbl)

        # add the widget to the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self._model_group)
        main_layout.addWidget(self._layer_group)
        main_layout.addWidget(self._automatic_seg_group)
        main_layout.addWidget(self._predictor_group)
        main_layout.addWidget(_info_group)

        # connections
        self._viewer.layers.events.changed.connect(self._on_layers_changed)
        self._viewer.layers.events.inserted.connect(self._on_layers_changed)
        self._viewer.layers.events.removed.connect(self._on_layers_changed)

    def _enable(self, state: bool) -> None:
        """Enable or disable the widget."""
        self._model_group.setEnabled(state)
        self._layer_group.setEnabled(state)
        self._automatic_seg_group.setEnabled(state)
        self._predictor_group.setEnabled(state)

    # ========================MODEL=========================

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

        self._load_worker(model_checkpoint, model_type)

    def _load_worker(self, model_checkpoint: str, model_type: str) -> None:
        self._load_info_lbl.setText("Loading model...")
        create_worker(
            self._load,
            model_checkpoint=model_checkpoint,
            model_type=model_type,
            _start_thread=True,
            _connect={"yielded": self._update_info},
        )

    def _update_info(self, args: tuple[bool, str]) -> None:
        """Update the info label."""
        loaded, device = args
        if loaded:
            _loaded_status = f"Model loaded successfully.\nUsing: {device.upper()}"
            self._model_status_label.setText("Model loaded successfully.")
        else:
            _loaded_status = "Error while loading model!"
            self._model_status_label.setText("Model not loaded.")
            self._sam = None
            self._predictor = None

        self._load_info_lbl.setText(_loaded_status)

        if self._console:
            self._console.push({"sam": self._sam, "predictor": self._predictor})

    def _load(
        self, model_checkpoint: str, model_type: str
    ) -> Generator[tuple[bool, str], None, None]:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
        except Exception as e:
            self._sam = None
            yield False, ""
            print(e)

        self._sam.to(device=device)

        try:
            self._predictor = SamPredictor(self._sam)
        except Exception as e:
            self._predictor = None
            yield False, ""
            print(e)

        yield True, device

    # ========================LAYER=========================

    def _on_layers_changed(self) -> None:
        """Update the layer combo box."""
        current_layer = self._image_combo.currentText()
        self._image_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._image_combo.addItem(layer.name)
        if current_layer and current_layer in self._viewer.layers:
            self._image_combo.setCurrentText(current_layer)

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

    # ====================AUTO MASK GENERATOR====================

    def _on_generate(self) -> None:
        """Start the mask generation."""
        self._info_lbl.setText("Generating masks...")

        self._init_generator()

        if self._mask_generator is None:
            return

        layer_name = self._image_combo.currentText()

        if (
            not self._viewer.layers
            or not layer_name
            or layer_name not in self._viewer.layers
        ):
            return

        image = self._convert_image_to_8bit(layer_name)

        self._generate_worker(image, layer_name)

    def _generate_worker(self, image: np.ndarray, layer_name: str) -> None:
        self._enable(False)
        create_worker(
            self._generate,
            image=image,
            layer_name=layer_name,
            _start_thread=True,
            _connect={
                "yielded": self._display_labels_auto_segmentation,
                "finished": self._on_auto_mask_generator_finished,
            },
        )

    def _on_auto_mask_generator_finished(self) -> None:
        """Enable the widget after the prediction is finished."""
        self._enable(True)
        if self._success:
            self._info_lbl.setText("Automatic Mask Generator finished.")
        else:
            self._info_lbl.setText("Error while running the Automatic Mask Generator!")

    def _init_generator(self) -> None:
        """Initialize the SAM Automatic Mask Generator."""
        self._mask_generator = None
        try:
            self._mask_generator = SamAutomaticMaskGenerator(
                model=self._sam,
                points_per_side=self._points_per_side.value(),
                points_per_batch=self._points_per_batch.value(),
                pred_iou_thresh=self._pred_iou_thresh.value(),
                stability_score_thresh=self._stability_score_thresh.value(),
                stability_score_offset=self._stability_score_offset.value(),
                box_nms_thresh=self._box_nms_thresh.value(),
                crop_n_layers=self._crop_n_layers.value(),
                crop_nms_thresh=self._crop_nms_thresh.value(),
                crop_overlap_ratio=self._crop_overlap_ratio.value(),
                crop_n_points_downscale_factor=self._crop_n_points_downscale_factor.value(),
                point_grids=None,
                min_mask_region_area=self._min_mask_region_area.value(),
                output_mode=self._output_mode.text(),
            )
            self._success = True
        except Exception as e:
            self._mask_generator = None
            self._success = False
            print(e)

    def _generate(
        self, image: np.ndarray, layer_name: str
    ) -> Generator[tuple[list[dict[str, Any]], str], None, None]:
        """Generate masks using the SAM Automatic Mask Generator."""
        try:
            masks = self._mask_generator.generate(image)
            self._success = True
        except Exception as e:
            self._success = False
            print(e)
            yield [], layer_name
            return
        yield masks, layer_name

    @ensure_main_thread
    def _display_labels_auto_segmentation(
        self, args: tuple[list[dict[str, Any]], str]
    ) -> None:
        """Display the masks in a stack."""
        masks, layer_name = args

        if self._console:
            self._console.push({"masks": masks})

        segmented: list[np.ndarray] = [
            mask["segmentation"]
            for mask in masks
            if (
                mask["area"] >= self._min_area.value()
                and mask["area"] <= self._max_area.value()
            )
        ]
        name = f"{layer_name}_masks[Automatic]"
        # # create a stack
        # stack = np.stack(segmented, axis=0)
        # self._viewer.add_image(stack, name=name, blending="additive")

        name = f"{layer_name}_labels[Automatic]"
        final_mask = np.zeros_like(segmented[0], dtype=np.int32)
        for mask in segmented:
            labeled_mask = measure.label(mask)
            labeled_mask[labeled_mask != 0] += final_mask.max()
            final_mask += labeled_mask
        self._viewer.add_labels(final_mask, name=name)

    # ==========================PREDCITOR===========================

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
                edge_color="magenta",
                face_color="magenta",
            )

        if (layer, 1) not in layers_meta:
            self._viewer.add_points(
                name=f"{layer}_points [FOREGROUND]",
                ndim=2,
                metadata={"id": layer, "type": 1},
                edge_color="green",
                face_color="green",
            )

    def _on_predict(self) -> None:
        """Start the prediction."""
        if self._predictor is None:
            return

        self._info_lbl.setText("Running Predictor...")

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

        self._predict_worker(layer_name, frg_points, bkg_points)

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

    def _predict_worker(
        self,
        layer_name: str,
        foreground_points: list[tuple[tuple[int, int], int]],
        background_points: list[tuple[tuple[int, int], int]],
    ) -> None:
        """Run the prediction in another thread."""
        self._enable(False)
        create_worker(
            self._predict,
            layer_name=layer_name,
            foreground_points=foreground_points,
            background_points=background_points,
            _start_thread=True,
            _connect={
                "yielded": self._display_labels_predictor,
                "finished": self._on_predict_finished,
            },
        )

    def _on_predict_finished(self) -> None:
        """Enable the widget after the prediction is finished."""
        self._enable(True)
        if self._success:
            self._info_lbl.setText("Predictor finished.")
        else:
            self._info_lbl.setText("Error while running the Predictor!")

    def _predict(
        self,
        layer_name: str,
        foreground_points: list[tuple[tuple[int, int], int]],
        background_points: list[tuple[tuple[int, int], int]],
    ) -> Generator[tuple[str, list[np.ndarray], list[float], bool], None, None]:
        """Run the SamPredictor."""
        try:
            image = self._convert_image_to_8bit(layer_name)
            self._predictor.set_image(image)

            if self._standard_radio.isChecked():
                masks, scores = self._standard_predictor(
                    foreground_points, background_points
                )
            else:
                masks, scores = self._loop_predictor(foreground_points)

            self._success = True

            yield layer_name, masks, scores, self._standard_radio.isChecked()
        except Exception as e:
            self._success = False
            print(e)

    def _standard_predictor(
        self,
        foreground_points: list[tuple[tuple[int, int], int]],
        background_points: list[tuple[tuple[int, int], int]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """The Standard SAM Predictor.

        Feed foreground and background points to the predictor in a list and get the
        masks and scores.
        """
        try:
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
            self._success = True
        except Exception as e:
            self._success = False
            print(e)
            masks = []
            score = []

        return masks, score

    def _loop_predictor(
        self, foreground_points: list[tuple[tuple[int, int], int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """The Loop SAM Predictor.

        Feed each foreground point to the predictor individually and get the masks and
        scores for each point.
        """
        try:
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
            self._success = True
        except Exception as e:
            self._success = False
            print(e)
            masks = []
            scores = []

        return masks, scores

    @ensure_main_thread
    def _display_labels_predictor(
        self, args: tuple[str, list[np.ndarray], list[float], bool]
    ) -> None:
        """Display the masks as labels in the viewer."""
        layer_name, masks, scores, standard = args

        if self._console:
            self._console.push({"masks": masks, "scores": scores})

        _type = "Standard" if standard else "Loop"
        name = f"{layer_name}_labels[{_type}]"

        if len(masks) == 1:
            labeled_mask = measure.label(masks[0])
            self._viewer.add_labels(labeled_mask, name=name)
            return

        final_mask = np.zeros_like(masks[0], dtype=np.int32)
        for mask in masks:
            labeled_mask = measure.label(mask)
            labeled_mask[labeled_mask != 0] += final_mask.max()
            final_mask += labeled_mask
        self._viewer.add_labels(final_mask, name=name)
