from __future__ import annotations

import contextlib
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
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from skimage import measure
from superqt.utils import create_worker, ensure_main_thread

from napari_sam_prompt._sub_widgets._auto_mask_generator import AutoMaskGeneratorWidget
from napari_sam_prompt._sub_widgets._predictor_widget import (
    BOXES,
    POINTS,
    PredictorWidget,
)

if TYPE_CHECKING:
    from napari.utils.events import Event
    from segment_anything.modeling import Sam


FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
EXTENDED = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

AUTO_MASK = "auto_mask_generator"
PREDICTOR = "predictor"
IMAGE_SET = "image_set"
EMBEDDINGS = "embeddings"
MASKS = "masks"
SCORES = "scores"
COORDS = "coords"

logging.basicConfig(
    # filename="napari_sam_prompt.log", # uncomment to log to a file in this directory
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ImageToPredictorMessageBox(QMessageBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setIcon(QMessageBox.Icon.Information)
        self.setWindowTitle("Image to Predictor")
        self.setText("Wait for the image to be set to the predictor...")
        self.setStandardButtons(QMessageBox.StandardButton.NoButton)


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

        self._image_to_predictor_msg = ImageToPredictorMessageBox(
            self._viewer.window._qt_viewer
        )

        self._device: str = ""
        self._sam: Sam | None = None
        self._predictor: SamPredictor | None = None
        self._mask_generator: SamAutomaticMaskGenerator | None = None
        self._current_image: str = ""

        self._setting_image: bool = False

        # this is to store all the info form the automatic mask generator and the
        # predictor. This is added to the napari console so we can check the info
        # TODO: maybe use dataclass
        self._stored_info: dict[str, dict[str, Any]] = {}
        # =========================STRUCTURE==========================
        # self._stored_info = {
        #     "image_name": {
        #         AUTO_MASK: list[dict[str, Any]],
        #         PREDICTOR: {
        #             IMAGE_SET: bool,
        #             EMBEDDINGS: torch.Tensor,
        #             POINTS: {
        #                 MASKS: list[np.ndarray],
        #                 SCORES: list[float],
        #                 COORDS: list[tuple[int, int]],
        #             },
        #             BOXES: {
        #                 MASKS: list[np.ndarray],
        #                 SCORES: list[float],
        #                 COORDS: list[tuple[int, int, int, int]],
        #             },
        #         },
        #     }
        # }
        # ============================================================

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

        self._viewer.layers.selection.events.changed.connect(self._on_layer_selected)

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

    def _on_image_combo_changed(self) -> None:
        """Update the current image."""
        self._current_image = self._image_combo.currentText()

    def _on_layers_changed(self, e: Event) -> None:
        """Update the layer combo box."""
        image_layers = [
            layer.name
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

        # remove the layers that are not in the viewer anymore
        for layer in list(self._stored_info.keys()):
            if layer in image_layers:
                continue
            self._stored_info.pop(layer, None)
            # update info in the console
            if self._console:
                self._console.push({"info": self._stored_info})
            # delete any associated layers
            for _layer in list(self._viewer.layers):
                if _layer.metadata.get("id") == layer:
                    self._viewer.layers.remove(_layer)

        # reverse the layers to show the last added layer first
        self._image_combo.clear()
        self._image_combo.addItems(reversed(image_layers))

        # add any newely added layers to the _stored_info
        for layer in image_layers:
            if layer not in self._stored_info:
                # self._image_set[layer] = (False, None)
                self._stored_info[layer] = {
                    AUTO_MASK: [],
                    PREDICTOR: {
                        IMAGE_SET: False,
                        EMBEDDINGS: None,
                        POINTS: {MASKS: [], SCORES: [], COORDS: []},
                        BOXES: {MASKS: [], SCORES: [], COORDS: []},
                    },
                }

    def _on_layer_selected(self, e: Event) -> None:
        """Change the current image to the selected layer and update the combo box."""
        with contextlib.suppress(Exception):
            active_layer = cast(napari.layers, e.source.active)
            layer_name = active_layer.name

            # in case we select a points or shapes layer
            if _id := active_layer.metadata.get("id"):
                layer_name = _id

            if self._current_image != layer_name:
                self._current_image = layer_name
                self._image_combo.setCurrentText(layer_name)

    def _convert_image(self, layer_name: str) -> np.ndarray:
        """Convert the image to 8-bit and stack to 3 channels."""
        # TODO: Handle already 8-bit, rgb images + stacks
        layer = cast(napari.layers.Image, self._viewer.layers[layer_name])
        data = layer.data
        # Normalize to the range 0-1
        data_three_channels = self._convert_to_three_channels(data)
        data_8bit = self._convert_8bit(data_three_channels)
        # Stack the image three times to create a 3-channel image

        return data_8bit.astype("uint8")

    def _convert_8bit(self, data: np.ndarray) -> np.ndarray:
        """Convert image to 8-bit."""
        for i in range(3):
            temp_data = data[:, :, i]
            data[:, :, i] = (
                255
                * ((temp_data - temp_data.min()) / (temp_data.max() - temp_data.min()))
            ).astype("uint8")
        return data

    def _convert_to_three_channels(self, data: np.ndarray) -> np.ndarray:
        """Convert the image to 3 channels."""
        channels = data.shape[-1] if len(data.shape) == 3 else 1
        if channels == 1:
            return np.stack([data] * 3, axis=-1)
        elif channels == 2:
            avg_channels = np.mean(data, axis=-1, keepdims=True)
            return np.concatenate([data, avg_channels], axis=-1)
        elif channels == 3:
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
            self._console.push({"sam": self._sam, PREDICTOR: self._predictor})

    def _load(
        self, model_checkpoint: str, model_type: str
    ) -> Generator[bool, None, None]:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
        except Exception as e:  # be more specific
            self._sam = None
            yield False
            logging.exception(e)
            return

        self._sam.to(device=self._device)

        try:
            self._predictor = SamPredictor(self._sam)
        except Exception as e:  # be more specific
            self._predictor = None
            yield False
            logging.exception(e)
            return

        yield True

    # ====================AUTO MASK GENERATOR====================

    def _on_generate(self) -> None:
        """Start the mask generation."""
        if self._sam is None:
            self._info_lbl.setText("Load a SAM model first.")
            return

        if (
            not self._viewer.layers
            or not self._current_image
            or self._current_image not in self._viewer.layers
        ):
            self._info_lbl.setText("No image layer selected.")
            return

        self._info_lbl.setText("Generating masks...")
        logging.info("Generating masks...")

        try:
            self._init_generator()
        except Exception as e:  # be more specific
            self._mask_generator = None
            msg = "Error while initializing the Automatic Mask Generator!"
            self._info_lbl.setText(msg)
            logging.exception(e)

        image = self._convert_image(self._current_image)

        self._generate_worker(image)

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

    def _generate_worker(self, image: np.ndarray) -> None:
        self._enable_all(False)
        create_worker(
            self._generate,
            image=image,
            _start_thread=True,
            _connect={
                "yielded": self._display_labels_auto_segmentation,
                "finished": self._on_auto_mask_generator_finished,
            },
        )

    def _generate(self, image: np.ndarray) -> Generator[list[np.ndarray], None, None]:
        """Generate masks using the SAM Automatic Mask Generator."""
        try:
            self._mask_generator = cast(SamAutomaticMaskGenerator, self._mask_generator)
            masks = self._mask_generator.generate(image)
            self._stored_info[self._current_image][AUTO_MASK] = masks
            self._success = True
        except Exception as e:  # be more specific
            self._success = False
            logging.exception(e)
            self._stored_info[self._current_image][AUTO_MASK] = []
            yield []
            return
        yield masks

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
    def _display_labels_auto_segmentation(self, masks: list[np.ndarray]) -> None:
        """Display the masks in a stack."""
        layer_name = self._current_image

        if self._console:
            self._console.push({"info": self._stored_info})

        segmented: list[np.ndarray] = [
            mask["segmentation"]
            for mask in masks
            if (
                mask["area"] >= self._automatic_seg_group._min_area.value()
                and mask["area"] <= self._automatic_seg_group._max_area.value()
            )
        ]

        # name = f"{layer_name}_masks[Automatic]"
        # # create a stack
        # stack = np.stack(segmented, axis=0)
        # self._viewer.add_image(stack, name=name, blending="additive")

        name = f"{layer_name}_labels[Automatic]"
        final_mask = np.zeros_like(segmented[0], dtype=np.int32)
        for mask in segmented:
            labeled_mask = measure.label(mask)
            labeled_mask[labeled_mask != 0] += final_mask.max()
            final_mask += labeled_mask

        try:
            labels = cast(napari.layers.Labels, self._viewer.layers[name])
            labels.data = final_mask
        except KeyError:
            self._viewer.add_labels(final_mask, name=name, metadata={"id": layer_name})

    # ====================SET IMAGE TO PREDICTOR====================

    def _prepare_and_run_predictor(self, layer: napari.layers) -> None:
        """Prepare the predictor and run it on the current image."""
        if self._predictor is None:
            self._info_lbl.setText("Load a SAM model first.")
            return

        # if self._current_image not in self._image_set:
        if self._current_image not in self._stored_info:
            self._info_lbl.setText("No image layer selected.")
            return

        predictor_info = self._stored_info[self._current_image][PREDICTOR]
        image_set = predictor_info.get(IMAGE_SET, False)
        embeddings = predictor_info.get(EMBEDDINGS)

        self._enable_all(False)

        # if the image have never been set to the predictor, set it in another thread
        # and when finished, run the predictor
        if not image_set or embeddings is None:
            self._setting_image = True
            self._image_to_predictor_msg.show()
            msg = "Setting the image to the predictor..."
            self._info_lbl.setText(msg)
            logging.info(msg)
            create_worker(
                self._image_to_predictor,
                _start_thread=True,
                _connect={"finished": lambda: self._on_image_set_finished(layer)},
            )

        # if the image has been set to the predictor but the embeddings are different
        # set the embeddings to the predictor from the _image_set variable
        elif not torch.allclose(self._predictor.features, embeddings):
            msg = "Setting the image embeddings to the predictor..."
            self._info_lbl.setText(msg)
            logging.info(msg)
            self._set_image_embeddings_to_predictor(image_set, embeddings)
            prompts = layer.data
            self._on_predict(self._predictor_widget.mode(), prompts)

        else:
            prompts = layer.data
            self._on_predict(self._predictor_widget.mode(), prompts)

    def _image_to_predictor(self) -> None:
        """Set the image to the predictor."""
        try:
            image = self._convert_image(self._current_image)
            self._predictor = cast(SamPredictor, self._predictor)
            self._predictor.set_image(image)

            # store the image embeddings
            store = self._stored_info[self._current_image][PREDICTOR]
            store[IMAGE_SET] = True
            store[EMBEDDINGS] = self._predictor.get_image_embedding()
            self._stored_info[self._current_image][PREDICTOR] = store

        except Exception as e:  # be more specific
            logging.exception(e)
            return

    def _on_image_set_finished(self, layer: napari.layers) -> None:
        """Enable the widget after setting the image to the predictor."""
        self._enable_all(True)

        self._setting_image = False

        self._image_to_predictor_msg.hide()

        predictor_info = self._stored_info[self._current_image][PREDICTOR]
        image_set = predictor_info.get(IMAGE_SET)
        embeddings = predictor_info.get(EMBEDDINGS)
        if image_set and embeddings is not None:
            msg = "Image set to the predictor."
            self._info_lbl.setText(msg)
            logging.info(msg)
            prompts = layer.data
            self._on_predict(self._predictor_widget.mode(), prompts)
        else:
            msg = "Error while setting the image to the predictor!"
            self._info_lbl.setText(msg)
            logging.error(msg)

    def _set_image_embeddings_to_predictor(
        self, image_set: bool, embeddings: torch.Tensor
    ) -> bool:
        """Set the image embeddings to the predictor."""
        if self._predictor is None:
            self._info_lbl.setText("Load a SAM model first.")
            return False
        if image_set and embeddings is not None:
            self._predictor.features = embeddings
            return True
        return False

    # ==========================PREDCITOR===========================

    def _add_layers(self, mode: str) -> None:
        """Add the layers for the correct prompt type."""
        if not self._current_image:
            return

        layers_meta = [
            (lay.metadata.get("prompt"), lay.metadata.get("id"))
            for lay in self._viewer.layers
        ]

        if mode == POINTS and (POINTS, self._current_image) not in layers_meta:
            layer = self._viewer.add_points(
                name=f"{self._current_image} [{POINTS.upper()}]",
                ndim=2,
                metadata={"prompt": POINTS, "id": self._current_image},
                edge_color="green",
                face_color="green",
            )
            layer.mode = "add"
            layer.events.data.connect(self._data_changed)

        elif mode == BOXES and (BOXES, self._current_image) not in layers_meta:
            layer = self._viewer.add_shapes(
                name=f"{self._current_image} [{BOXES.upper()}]",
                ndim=2,
                metadata={"prompt": BOXES, "id": self._current_image},
                face_color="white",
                edge_color="green",
                edge_width=3,
                opacity=0.4,
                blending="translucent",
            )
            layer.mode = "add_rectangle"
            layer.events.data.connect(self._data_changed)

    def _data_changed(self, event: Event) -> None:
        """Handle the data change event and run the predictor."""
        # return if the image is being set to the predictor
        if self._setting_image:
            return

        layer = cast(napari.layers, event.source)

        # clear the prompt layer if predictor is not set
        if self._predictor is None:
            # block to avoid recursion
            with layer.events.data.blocker():
                layer.data = []
            self._info_lbl.setText("Load a SAM model first.")
            return

        # stored = self._prompts_coords.get(self._current_image)
        stored = self._stored_info.get(self._current_image, {})
        mode = self._predictor_widget.mode()
        coords = stored.get(PREDICTOR, {}).get(mode, {}).get(COORDS, [])

        # if a prompt is removed
        if len(layer.data) < len(coords):
            name = f"{self._current_image}_LABELS [{mode.upper()}]"
            if len(layer.data) == 0:
                # no prompts are left
                self._clear_mode_info(mode)
                # delete the labels layer
                with contextlib.suppress(KeyError):
                    labels = cast(napari.layers.Labels, self._viewer.layers[name])
                    self._viewer.layers.remove(labels)

            else:
                # if the use removed a prompt, but we still have some prompts left
                # we need to re-run the predictor with the new prompts.

                # if there is only one point, we need to clear the stored prompts
                if len(layer.data) == 1:
                    self._clear_mode_info(mode)

                # clearing the labels layer
                with contextlib.suppress(KeyError):
                    labels = cast(napari.layers.Labels, self._viewer.layers[name])
                    labels.data = np.zeros_like(labels.data)
                self._prepare_and_run_predictor(layer)

        elif len(layer.data) > len(coords):
            self._prepare_and_run_predictor(layer)

    def _clear_mode_info(self, mode: str) -> None:
        """Clear the stored info for the current mode."""
        self._stored_info[self._current_image][PREDICTOR][mode][COORDS] = []
        self._stored_info[self._current_image][PREDICTOR][mode][MASKS] = []
        self._stored_info[self._current_image][PREDICTOR][mode][SCORES] = []

    def _on_predict(self, mode: str, prompts: np.ndarray) -> None:
        """Prepare the prompts and run the predictor."""
        if self._sam is None or self._predictor is None:
            self._info_lbl.setText("Load a SAM model first.")
            return

        msg = f"Running Predictor with {mode} Prompts..."
        self._info_lbl.setText(msg)
        logging.info(msg)

        if mode == POINTS:
            prompts = [(int(point[1]), int(point[0])) for point in prompts]
        else:  # mode == BOXES:
            # prompt as need to be top_left and bottom_right coordinates
            boxes = []
            for box in prompts:
                top_left = (int(box[0][1]), int(box[0][0]))
                bottom_right = (int(box[2][1]), int(box[2][0]))
                boxes.append((*top_left, *bottom_right))
            prompts = boxes

        # if the current image has already prompts for the current mode and the length
        # of the new prompts is less than the existing prompts, means thet the user
        # removed a prompts, so we need re-run the predictor with the new prompts
        predictor_info = self._stored_info[self._current_image][PREDICTOR]
        if stored_prompts := predictor_info.get(mode, []):
            n_prompts = len(stored_prompts[COORDS])
            # use only the last added prompt
            updated_prompts = [prompts[-1]] if len(prompts) > n_prompts else prompts
        else:
            updated_prompts = prompts

        # update the prompts coordinates
        self._stored_info[self._current_image][PREDICTOR][mode][COORDS] = prompts

        # run the predictor
        self._predict_worker(mode, updated_prompts)

    def _predict_worker(self, mode: str, prompts: np.ndarray) -> None:
        """Run the prediction in another thread."""
        if self._predictor is None:
            return

        self._enable_all(False)

        create_worker(
            self._predict_with_prompts,
            mode=mode,
            prompts=prompts,
            _start_thread=True,
            _connect={
                "yielded": self._display_labels_predictor,
                "finished": self._on_predict_finished,
            },
        )

    def _predict_with_prompts(
        self, mode: str, prompts: np.ndarray
    ) -> Generator[tuple[list[np.ndarray], list[float], str], None, None]:
        try:
            self._predictor = cast(SamPredictor, self._predictor)

            store = self._stored_info[self._current_image][PREDICTOR][mode]

            if mode == POINTS:
                masks, scores = self._predict_with_points(prompts, store)
            elif mode == BOXES:
                masks, scores = self._predict_with_boxes(prompts, store)

            self._success = True

            yield masks, scores, mode

        except Exception as e:  # be more specific
            self._success = False
            logging.exception(e)
            yield [], [], ""

    def _predict_with_points(
        self, prompts: np.ndarray, store: dict
    ) -> tuple[list[np.ndarray], list[float]]:
        """Run the predictor with points as prompts.

        The difference between the if/else block is the way the prompts are passed to
        the predictor. In the first case, a single point is passed since it is the last
        point that was added. This will result in a faster run of the predictor. In the
        second case, all the points are passed to the predictor, because either all or
        some of the points were removed. So we need to re-run the predictor with the new
        prompts.
        """
        self._predictor = cast(SamPredictor, self._predictor)
        if len(prompts) == 1:
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(prompts),
                point_labels=np.array([1]),  # prompt is always 1 point
                multimask_output=False,
            )
            store[MASKS].append(masks)
            store[SCORES].append(scores)

        else:
            # should be triggered when the user removes prompt
            masks, scores = [], []
            for prompt in prompts:
                mask, score, _ = self._predictor.predict(
                    point_coords=np.array([prompt]),
                    point_labels=np.array([1]),  # prompt is always 1 point
                    multimask_output=False,
                )
                masks.append(mask)
                scores.append(score)
            store[MASKS] = masks
            store[SCORES] = scores

        self._stored_info[self._current_image][PREDICTOR][POINTS] = store

        return masks, scores

    def _predict_with_boxes(
        self, prompts: np.ndarray, store: dict
    ) -> tuple[np.ndarray, list[float]]:
        """Run the predictor with boxes as prompts.

        The difference between the if/else block is the way the prompts are passed to
        the predictor. In the first case, a single box is passed since it is the last
        box that was added. This will result in a faster run of the predictor. In the
        second case, all the boxes are passed to the predictor, because either all or
        some of the boxes were removed. So we need to re-run the predictor with the new
        prompts.
        """
        self._predictor = cast(SamPredictor, self._predictor)
        if len(prompts) == 1:
            masks, scores, _ = self._predictor.predict(
                box=np.array([prompts]), multimask_output=False
            )
            store[MASKS].append(masks)
            store[SCORES].append(scores)
        else:
            # should be triggered when the user removes prompt
            masks, scores = [], []
            for prompt in prompts:
                mask, score, _ = self._predictor.predict(
                    box=np.array([prompt]), multimask_output=False
                )
                masks.append(mask)
                scores.append(score)
            store[MASKS] = masks
            store[SCORES] = scores

        self._stored_info[self._current_image][PREDICTOR][BOXES] = store

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

        if self._console:
            self._console.push({"info": self._stored_info})

    @ensure_main_thread  # type: ignore [misc]
    def _display_labels_predictor(
        self, args: tuple[list[np.ndarray], list[float], str]
    ) -> None:
        """Display the masks as labels in the viewer."""
        masks, _, mode = args
        layer_name = self._current_image

        name = f"{layer_name}_LABELS [{mode.upper()}]"

        stored = self._stored_info.get(layer_name)
        if stored is None:
            # first time
            self._viewer.add_labels(masks, name=name, metadata={"id": layer_name})
        else:
            stored_masks = stored[PREDICTOR][mode][MASKS]

            if len(stored_masks) == 0:
                return

            mask_for_labels = self._masks_for_labels(stored_masks)
            try:
                labels = cast(napari.layers.Labels, self._viewer.layers[name])
                labels.data = mask_for_labels
            except KeyError:
                self._viewer.add_labels(
                    mask_for_labels, name=name, metadata={"id": layer_name}
                )

        # keep the prompt layer as the active layer
        prompt_layer = self._viewer.layers[f"{layer_name} [{mode.upper()}]"]
        self._viewer.layers.selection.active = prompt_layer

    def _masks_for_labels(self, masks: list[np.ndarray]) -> np.ndarray:
        """Create the mask data to be used in the labels layer."""
        final_mask = np.zeros_like(masks[0], dtype=np.int32)
        for mask in masks:
            labeled_mask = measure.label(mask)
            labeled_mask[labeled_mask != 0] += final_mask.max()
            final_mask += labeled_mask
        return final_mask
