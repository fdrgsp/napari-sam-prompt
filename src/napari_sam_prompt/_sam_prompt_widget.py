from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generator, cast

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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from skimage import measure
from superqt.utils import create_worker, ensure_main_thread, signals_blocked

from napari_sam_prompt._sub_widgets._auto_mask_generator import AutoMaskGeneratorWidget
from napari_sam_prompt._sub_widgets._predictor_widget import PredictorWidget

if TYPE_CHECKING:
    from segment_anything.modeling import Sam

POINTS = "points"
BOXES = "boxes"
STANDARD = "standard"
LOOP = "loop"
FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
EXTENDED = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

logging.basicConfig(
    # filename="napari_sam_prompt.log", # uncomment to log to a file in this directory
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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

        self._device: str = ""
        self._sam: Sam | None = None
        self._predictor: SamPredictor | None = None
        self._mask_generator: SamAutomaticMaskGenerator | None = None
        self._image_set: bool = False

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
        self._load_module_btn = QPushButton("Load Selected Model")
        self._load_module_btn.clicked.connect(self._on_load)

        _model_group_layout.addWidget(_model_lbl, 0, 0)
        _model_group_layout.addWidget(self._model_le, 0, 1)
        _model_group_layout.addWidget(self._model_browse_btn, 0, 2)
        _model_group_layout.addWidget(_model_type_lbl, 1, 0)
        _model_group_layout.addWidget(self._model_type_le, 1, 1, 1, 2)
        _model_group_layout.addWidget(self._load_module_btn, 2, 0, 1, 3)

        # add layer selector groupbox
        self._layer_group = QGroupBox("Layer Selector")
        _image_group_layout = QGridLayout(self._layer_group)
        _image_combo_lbl = QLabel("Layer:")
        _image_combo_lbl.setSizePolicy(FIXED)
        self._image_combo = QComboBox()
        self._image_combo.currentTextChanged.connect(self._on_image_combo_changed)
        _image_group_layout.addWidget(_image_combo_lbl, 0, 0)
        _image_group_layout.addWidget(self._image_combo, 0, 1)

        # add automatic segmentation
        self._automatic_seg_group = AutoMaskGeneratorWidget()
        self._automatic_seg_group.generateSignal.connect(self._on_generate)

        # add predictor widget
        self._predictor_widget = PredictorWidget()
        self._predictor_widget.predictSignal.connect(self._on_predict)
        self._predictor_widget.addLayersSignal.connect(self._add_layers)

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
        main_layout.addWidget(self._predictor_widget)
        main_layout.addWidget(_info_group)
        main_layout.addStretch()

        # viewer connections
        self._viewer.layers.events.changed.connect(self._on_layers_changed)
        self._viewer.layers.events.inserted.connect(self._on_layers_changed)
        self._viewer.layers.events.removed.connect(self._on_layers_changed)

        self._enable_widgets(False)

    def _enable_widgets(self, state: bool) -> None:
        """Enable or disable the widget."""
        self._layer_group.setEnabled(state)
        self._automatic_seg_group.setEnabled(state)
        self._predictor_widget.setEnabled(state)

    def _enable_all(self, state: bool) -> None:
        """Enable or disable the widget."""
        self._model_group.setEnabled(state)
        self._enable_widgets(state)

    def _on_layers_changed(self) -> None:
        """Update the layer combo box."""
        current_combo_layers = [
            self._image_combo.itemText(i) for i in range(self._image_combo.count())
        ]

        # block signals to avoid setting the image to the predictor while updating
        with signals_blocked(self._image_combo):
            self._image_combo.clear()
            layers = [
                layer.name
                for layer in self._viewer.layers
                if isinstance(layer, napari.layers.Image)
            ]
            self._image_combo.addItems(layers)

        if not layers:
            self._image_set = False
            return

        if layers != current_combo_layers:
            # set the image to the predictor
            self._on_image_combo_changed()

    def _convert_image(self, layer_name: str) -> np.ndarray:
        """Convert the image to 8-bit and stack to 3 channels."""
        # TODO: Handle already 8-bit, rgb images + stacks
        layer = cast(napari.layers.Image, self._viewer.layers[layer_name])
        data = layer.data
        # Normalize to the range 0-1
        img_8bit = self._convert_8bit(data)
        # Stack the image three times to create a 3-channel image
        img = self._convert_to_three_channels(img_8bit)

        return img
    
    def _convert_8bit(self, data):
        '''Convert image to 8-bit'''
        return (255*(data-data.min())/(data.max()-data.min())).astype('uint8')
    
    def _convert_to_three_channels(self, data):
        '''Convert image to 3-channel image'''
        if len(data.shape)==3:
            channels = data.shape[-1]
        else:
            channels = 1

        if channels==1:
            return np.stack([image] * 3, axis=-1)
        elif channels==2:
            avg_channels = np.mean(data, axis=-1, keepdims=True)
            return np.concatenate([data, avg_channels], axis=-1)
        elif channels==3:
            return data
        else:
            avg_channels = np.mean(data, axis=-1, keepdims=True)
            return np.stack([avg_channels] * 3, axis=-1)
    



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

        self._load_info_lbl.setText("Loading model...")
        logging.info("Loading model...")

        self._load_worker(model_checkpoint, model_type)

    def _load_worker(self, model_checkpoint: str, model_type: str) -> None:
        self._info_lbl.setText("")
        create_worker(
            self._load,
            model_checkpoint=model_checkpoint,
            model_type=model_type,
            _start_thread=True,
            _connect={"yielded": self._update_info},
        )

    def _update_info(self, loaded: bool) -> None:
        """Update the info label."""
        if loaded:
            self._enable_widgets(True)
            _loaded_status = f"Model loaded successfully.\nUsing: {self._device}"
            logging.info("Model loaded successfully.")
        else:
            self._enable_widgets(False)
            _loaded_status = "Error while loading model!"
            self._sam = None
            self._predictor = None
            logging.error("Error while loading model!")

        self._load_info_lbl.setText(_loaded_status)

        if self._console:
            self._console.push({"sam": self._sam, "predictor": self._predictor})

    def _load(
        self, model_checkpoint: str, model_type: str
    ) -> Generator[bool, None, None]:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
        except Exception as e:
            self._sam = None
            yield False
            logging.exception(e)
            return

        self._sam.to(device=self._device)

        try:
            self._predictor = SamPredictor(self._sam)
        except Exception as e:
            self._predictor = None
            yield False
            logging.exception(e)
            return

        yield True

    # ====================SET IMAGE TO PREDICTOR====================

    def _on_image_combo_changed(self) -> None:
        """Set the image to the predictor."""
        if self._predictor is None:
            self._info_lbl.setText("Load a SAM model first.")
            return
        self._enable_widgets(False)
        self._set_image_to_predictor_worker()

    def _set_image_to_predictor_worker(self) -> None:
        """Set the image to the predictor in another thread."""
        create_worker(
            self._set_image_to_predictor,
            _start_thread=True,
            _connect={"finished": self._on_image_set_finished},
        )

    def _set_image_to_predictor(self) -> None:
        """Set the image to the predictor."""
        self._image_set = False
        msg = "Setting the image to the predictor..."
        self._info_lbl.setText(msg)
        logging.info(msg)

        try:
            image = self._convert_image(self._image_combo.currentText())
            self._predictor = cast(SamPredictor, self._predictor)
            self._predictor.set_image(image)
            self._image_set = True
        except Exception as e:
            logging.exception(e)
            return

    def _on_image_set_finished(self) -> None:
        """Enable the widget after setting the image to the predictor."""
        self._enable_widgets(True)
        if self._image_set:
            self._info_lbl.setText("Image set to the predictor.")
            logging.info("Image set to the predictor.")
        else:
            self._info_lbl.setText("Error while setting the image to the predictor!")
            logging.error("Error while setting the image to the predictor!")

    # ====================AUTO MASK GENERATOR====================

    def _on_generate(self) -> None:
        """Start the mask generation."""
        if self._sam is None:
            self._info_lbl.setText("Load a SAM model first.")
            return

        layer_name = self._image_combo.currentText()

        if (
            not self._viewer.layers
            or not layer_name
            or layer_name not in self._viewer.layers
        ):
            self._info_lbl.setText("No image layer selected.")
            return

        self._info_lbl.setText("Generating masks...")
        logging.info("Generating masks...")

        try:
            self._init_generator()
        except Exception as e:
            self._mask_generator = None
            self._info_lbl.setText(
                "Error while initializing the Automatic Mask Generator!"
            )
            logging.exception(e)

        image = self._convert_image(layer_name)

        self._generate_worker(image, layer_name)

    def _init_generator(self) -> None:
        """Initialize the SAM Automatic Mask Generator."""
        self._mask_generator = None
        options = self._automatic_seg_group
        self._mask_generator = SamAutomaticMaskGenerator(
            model=self._sam,
            points_per_side=options._points_per_side.value(),
            points_per_batch=options._points_per_batch.value(),
            pred_iou_thresh=options._pred_iou_thresh.value(),
            stability_score_thresh=options._stability_score_thresh.value(),
            stability_score_offset=options._stability_score_offset.value(),
            box_nms_thresh=options._box_nms_thresh.value(),
            crop_n_layers=options._crop_n_layers.value(),
            crop_nms_thresh=options._crop_nms_thresh.value(),
            crop_overlap_ratio=options._crop_overlap_ratio.value(),
            crop_n_points_downscale_factor=options._crop_n_points_downscale_factor.value(),
            point_grids=None,
            min_mask_region_area=options._min_mask_region_area.value(),
            output_mode=options._output_mode.text(),
        )

    def _generate_worker(self, image: np.ndarray, layer_name: str) -> None:
        self._enable_all(False)
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

    def _generate(
        self, image: np.ndarray, layer_name: str
    ) -> Generator[tuple[list[dict[str, Any]], str], None, None]:
        """Generate masks using the SAM Automatic Mask Generator."""
        try:
            self._mask_generator = cast(SamAutomaticMaskGenerator, self._mask_generator)
            masks = self._mask_generator.generate(image)
            self._success = True
        except Exception as e:
            self._success = False
            logging.exception(e)
            yield [], layer_name
            return
        yield masks, layer_name

    def _on_auto_mask_generator_finished(self) -> None:
        """Enable the widget after the prediction is finished."""
        self._enable_all(True)
        if self._success:
            self._info_lbl.setText("Automatic Mask Generator finished.")
            logging.info("Automatic Mask Generator finished.")
        else:
            self._info_lbl.setText("Error while running the Automatic Mask Generator!")
            logging.error("Error while running the Automatic Mask Generator!")

    @ensure_main_thread  # type: ignore [misc]
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
                mask["area"] >= self._automatic_seg_group._min_area.value()
                and mask["area"] <= self._automatic_seg_group._max_area.value()
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

    def _add_layers(self, value: dict) -> None:
        """Add the layers for the correct prompt type."""
        layer_name = self._image_combo.currentText()

        if not layer_name:
            return

        predictor_prompt = value["predictor_prompt"]

        if predictor_prompt == POINTS:
            self._add_points_layers(layer_name)
        elif predictor_prompt == BOXES:
            self._add_boxes_layers(layer_name)

    def _add_points_layers(self, layer_name: str) -> None:
        """Add the points layers to the viewer."""
        layers_meta = [
            (
                lay.metadata.get("prompt"),
                lay.metadata.get("id"),
                lay.metadata.get("type"),
            )
            for lay in self._viewer.layers
        ]

        if (POINTS, layer_name, 0) not in layers_meta:
            self._viewer.add_points(
                name=f"{layer_name}_points [BACKGROUND]",
                ndim=2,
                metadata={"prompt": POINTS, "id": layer_name, "type": 0},
                edge_color="magenta",
                face_color="magenta",
            ).mode = "add"

        if (POINTS, layer_name, 1) not in layers_meta:
            self._viewer.add_points(
                name=f"{layer_name}_points [FOREGROUND]",
                ndim=2,
                metadata={"prompt": POINTS, "id": layer_name, "type": 1},
                edge_color="green",
                face_color="green",
            ).mode = "add"

    def _add_boxes_layers(self, layer_name: str) -> None:
        """Add the boxshapeses layers to the viewer."""
        layers_meta = [
            (lay.metadata.get("prompt"), lay.metadata.get("id"))
            for lay in self._viewer.layers
        ]

        if (BOXES, layer_name) not in layers_meta:
            self._viewer.add_shapes(
                name=f"{layer_name}_boxes",
                ndim=2,
                metadata={"prompt": BOXES, "id": layer_name},
                face_color="white",
                edge_color="green",
                edge_width=3,
                opacity=0.4,
                blending="translucent",
            ).mode = "add_rectangle"

    def _on_predict(self, value: dict) -> None:
        if self._sam is None or self._predictor is None:
            self._info_lbl.setText("Load a SAM model first.")
            return

        layer_name = self._image_combo.currentText()

        if (
            not self._viewer.layers
            or not layer_name
            or layer_name not in self._viewer.layers
        ):
            self._info_lbl.setText("No image layer selected.")
            return

        predictor_prompt = value["predictor_prompt"]
        predictor_type = value["predictor_type"]

        if not self._image_set:
            self._on_image_combo_changed()

        msg = f"Running Predictor with {predictor_prompt} Prompt..."
        self._info_lbl.setText(msg)
        logging.info(msg)

        prompts: (
            tuple[list[tuple[tuple[int, int], int]], list[tuple[tuple[int, int], int]]]
            | list[tuple[int, int, int, int]]
            | None
        ) = None

        if predictor_prompt == POINTS:
            prompts = self._get_points_coordinates(layer_name)
            if prompts is None:
                msg = "No Foreground or Background points."
                logging.error(msg)
                self._info_lbl.setText(msg)
                return

        elif predictor_prompt == BOXES:
            prompts = self._get_boxes_coordinates(layer_name)
            if prompts is None:
                msg = "No shapes."
                logging.error(msg)
                self._info_lbl.setText(msg)
                return

        self._predict_worker(predictor_prompt, layer_name, prompts, predictor_type)

    def _get_points_coordinates(
        self, layer_name: str
    ) -> (
        tuple[list[tuple[tuple[int, int], int]], list[tuple[tuple[int, int], int]]]
        | None
    ):
        """Get the points coordinates from the Points layer."""
        frg_point_layer, bkg_point_layer = self._get_point_layers(layer_name)

        if frg_point_layer is None or bkg_point_layer is None:
            return None

        frg_points: list[tuple[tuple[int, int], int]] = []
        for p in frg_point_layer.data:
            x, y = p[1], p[0]
            frg_points.append(((x, y), 1))

        bkg_points: list[tuple[tuple[int, int], int]] = []
        for p in bkg_point_layer.data:
            x, y = int(p[1]), int(p[0])
            bkg_points.append(((x, y), 0))

        return frg_points, bkg_points

    def _get_point_layers(
        self, layer_name: str
    ) -> tuple[napari.layers.Points | None, napari.layers.Points | None]:
        """Get the correct points layers from the viewer."""
        frg_point_layer = None
        bkg_point_layer = None

        for layer in self._viewer.layers:
            if (
                isinstance(layer, napari.layers.Points)
                and layer.metadata.get("id") == layer_name
                and layer.metadata.get("prompt") == POINTS
            ):
                if layer.metadata.get("type") == 1:
                    frg_point_layer = layer
                elif layer.metadata.get("type") == 0:
                    bkg_point_layer = layer

        return frg_point_layer, bkg_point_layer

    def _get_boxes_coordinates(
        self, layer_name: str
    ) -> list[tuple[int, int, int, int]] | None:
        """Get the box coordinates from the Shapes layer."""
        boxes_layer = self._get_shapes_layers(layer_name)

        if boxes_layer is None:
            return None

        box_coords: list[tuple[int, int, int, int]] = []
        for box in boxes_layer.data:
            top_left, _, bottom_right, _ = box
            # round the coordinates
            top_left = (int(top_left[1]), int(top_left[0]))
            bottom_right = (int(bottom_right[1]), int(bottom_right[0]))
            box_coords.append((*top_left, *bottom_right))

        return box_coords or None

    def _get_shapes_layers(self, layer_name: str) -> napari.layers.Shapes | None:
        """Get the correct shapes layer from the viewer."""
        for layer in self._viewer.layers:
            if (
                isinstance(layer, napari.layers.Shapes)
                and layer.metadata.get("id") == layer_name
                and layer.metadata.get("prompt") == BOXES
            ):
                return layer
        return None

    def _predict_worker(
        self,
        predictor_prompt: str,  # POINTS | BOXES
        layer_name: str,
        prompts: (
            tuple[list[tuple[tuple[int, int], int]], list[tuple[tuple[int, int], int]]]
            | list[tuple[int, int, int, int]]
            | None
        ),
        predictor_type: str,  # STANDARD | LOOP
    ) -> None:
        """Run the prediction in another thread."""
        if prompts is None or self._predictor is None:
            return

        self._enable_all(False)

        if predictor_prompt == POINTS:
            foreground_points, background_points = prompts
            create_worker(
                self._predict_with_points,
                predictor_type=predictor_type,
                layer_name=layer_name,
                foreground_points=foreground_points,
                background_points=background_points,
                _start_thread=True,
                _connect={
                    "yielded": self._display_labels_predictor,
                    "finished": self._on_predict_finished,
                },
            )
        elif predictor_prompt == BOXES:
            create_worker(
                self._predict_with_boxes,
                predictor_type=predictor_type,
                layer_name=layer_name,
                boxes=prompts,
                _start_thread=True,
                _connect={
                    "yielded": self._display_labels_predictor,
                    "finished": self._on_predict_finished,
                },
            )

    def _predict_with_points(
        self,
        predictor_type: str,  # STANDARD | LOOP
        layer_name: str,
        foreground_points: list[tuple[tuple[int, int], int]],
        background_points: list[tuple[tuple[int, int], int]],
    ) -> Generator[tuple[str, list[np.ndarray], list[float], str], None, None]:
        """Run the SamPredictor."""
        try:
            # image = self._convert_image_to_8bit(layer_name)
            # self._predictor = cast(SamPredictor, self._predictor)
            # self._predictor.set_image(image)

            if predictor_type == STANDARD:
                masks, scores = self._standard_points_predictor(
                    foreground_points, background_points
                )
            elif predictor_type == LOOP:
                masks, scores = self._loop_points_predictor(foreground_points)

            self._success = True

            yield layer_name, masks, scores, predictor_type
        except Exception as e:
            self._success = False
            logging.exception(e)

    def _standard_points_predictor(
        self,
        foreground_points: list[tuple[tuple[int, int], int]],
        background_points: list[tuple[tuple[int, int], int]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """The Standard SAM Predictor.

        Feed foreground and background points to the predictor in a list and get the
        masks and scores.
        """
        try:
            self._predictor = cast(SamPredictor, self._predictor)
            input_point = [point for point, _ in foreground_points] + [
                bg_points for bg_points, _ in background_points
            ]
            input_label = [label for _, label in foreground_points] + [
                bg_label for _, bg_label in background_points
            ]

            masks, score, _ = self._predictor.predict(
                point_coords=np.array(input_point),
                point_labels=np.array(input_label),
                multimask_output=False,
            )
            self._success = True
        except Exception as e:
            self._success = False
            logging.exception(e)
            return [], []

        return masks, score

    def _loop_points_predictor(
        self, foreground_points: list[tuple[tuple[int, int], int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """The Loop SAM Predictor.

        Feed each foreground point to the predictor individually and get the masks and
        scores for each point.
        """
        try:
            self._predictor = cast(SamPredictor, self._predictor)
            masks: list[np.ndarray] = []
            scores: list[float] = []
            for point, label in foreground_points:
                mask, score, _ = self._predictor.predict(
                    point_coords=np.array([point]),
                    point_labels=np.array([label]),
                    multimask_output=False,
                )
                masks.append(mask)
                scores.append(score)
            self._success = True
        except Exception as e:
            self._success = False
            logging.exception(e)
            return [], []

        return masks, scores

    def _predict_with_boxes(
        self,
        predictor_type: str,  # STANDARD | LOOP
        layer_name: str,
        boxes: list[tuple[int, int, int, int]],
    ) -> Generator[tuple[str, list[np.ndarray], list[float], str], None, None]:
        """Run the SamPredictor."""
        try:
            # image = self._convert_image_to_8bit(layer_name)
            # self._predictor = cast(SamPredictor, self._predictor)
            # self._predictor.set_image(image)

            if predictor_type == LOOP:
                masks, scores = self._loop_boxes_predictor(boxes)

            self._success = True

            yield layer_name, masks, scores, predictor_type
        except Exception as e:
            self._success = False
            logging.exception(e)

    def _loop_boxes_predictor(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """The Standard SAM Predictor.

        Feed boxes to the predictor in a list and get the masks and scores.
        """
        try:
            self._predictor = cast(SamPredictor, self._predictor)
            masks: list[np.ndarray] = []
            scores: list[float] = []
            for box in boxes:
                mask, score, _ = self._predictor.predict(
                    box=np.array(box), multimask_output=False
                )
                masks.append(mask)
                scores.append(score)

            self._success = True
        except Exception as e:
            self._success = False
            logging.exception(e)
            return [], []

        return masks, scores

    def _on_predict_finished(self) -> None:
        """Enable the widget after the prediction is finished."""
        self._enable_all(True)
        if self._success:
            self._info_lbl.setText("Predictor finished.")
            logging.info("Predictor finished.")
        else:
            self._info_lbl.setText("Error while running the Predictor!")
            logging.error("Error while running the Predictor!")

    @ensure_main_thread  # type: ignore [misc]
    def _display_labels_predictor(
        self, args: tuple[str, list[np.ndarray], list[float], str]
    ) -> None:
        """Display the masks as labels in the viewer."""
        layer_name, masks, scores, predictor_type = args

        if self._console:
            self._console.push({"masks": masks, "scores": scores})

        name = f"{layer_name}_labels[{predictor_type}]"

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
