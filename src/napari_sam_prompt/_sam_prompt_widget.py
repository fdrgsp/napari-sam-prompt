from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Generator, cast

import napari.layers
import napari.viewer
import numpy as np
import torch
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from skimage import measure
from superqt.utils import create_worker, ensure_main_thread

from napari_sam_prompt._sub_widgets._auto_mask_generator_widget import (
    AutoMaskGeneratorWidget,
)
from napari_sam_prompt._sub_widgets._load_model_widget import LoadModelWidget
from napari_sam_prompt._sub_widgets._predictor_widget import (
    BOXES,
    POINTS,
    POINTS_FB,
    PredictorWidget,
)

from ._util import _convert_image

if TYPE_CHECKING:
    from napari.utils.events import Event
    from segment_anything.modeling import Sam


FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

AUTO_MASK = "auto_mask_generator"
PREDICTOR = "predictor"
IMAGE_SET = "image_set"
EMBEDDINGS = "embeddings"
MASKS = "masks"
SCORES = "scores"
COORDS = "coords"
FWD_BKG = "fwd_or_bkg"
GREEN = "green"
MAGENTA = "magenta"
MAGENTA_CODE = [1, 0, 1, 1]

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
        self._model_checkpoint: str = model_checkpoint
        self._model_type: str = model_type

        self._current_image: str = ""

        # this is to store all the info form the automatic mask generator and the
        # predictor. This is added to the napari console so we can check the info
        # TODO: maybe use dataclass
        self._stored_info: dict[str, dict[str, Any]] = {
            "model": {
                "model_checkpoint": self._model_checkpoint,
                "model_type": self._model_type,
            }
        }
        # =========================STRUCTURE==========================
        # self._stored_info = {
        #     "model": {
        #           "model_checkpoint": "path/to/model.pth",
        #           "model_type": "model_type",
        #     "image_name": {
        #         AUTO_MASK: list[dict[str, Any]],
        #         PREDICTOR: {
        #             IMAGE_SET: bool,
        #             EMBEDDINGS: torch.Tensor,
        #             POINTS: {
        #                 MASKS: list[np.ndarray],
        #                 SCORES: list[float],
        #                 COORDS: list[tuple[int, int]],
        #                 FWD_BKG: list[int],
        #             },
        #             POINTS_FB: {
        #                 MASKS: list[np.ndarray],
        #                 SCORES: list[float],
        #                 COORDS: list[tuple[int, int]],
        #                 FWD_BKG: list[int],
        #             },
        #             BOXES: {
        #                 MASKS: list[np.ndarray],
        #                 SCORES: list[float],
        #                 COORDS: list[tuple[int, int, int, int]],
        #                 FWD_BKG: list[int],
        #             },
        #         },
        #     }
        # }
        # ============================================================

        self._success = False

        # Add the model groupbox
        self._model_group = LoadModelWidget(
            model_checkpoint=self._model_checkpoint, model_type=self._model_type
        )
        self._model_group.loadSignal.connect(self._on_load)

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
        self._automatic_seg_group.filterSignal.connect(self._on_filter)

        # add predictor widget
        self._predictor_widget = PredictorWidget()
        self._predictor_widget.addLayersSignal.connect(self._add_layers)
        self._predictor_widget.predictSignal.connect(self._on_predict_signal)

        # info group
        _info_group = QGroupBox("Info")
        _info_group_layout = QVBoxLayout(_info_group)
        self._load_info_lbl = QLabel("Model not loaded.")
        self._info_lbl = QLabel()
        self._info_lbl.setWordWrap(True)
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

        if model_checkpoint and model_type:
            self._on_load()

        if self._console:
            self._console.push({"info": self._stored_info})

    def _message_to_log(self, msg: str) -> None:
        """Log the message and display it in the info label."""
        self._info_lbl.setText(msg)
        logging.info(msg)

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
        for key in list(self._stored_info.keys()):
            if key in image_layers or key == "model":
                continue
            self._stored_info.pop(key, None)
            # delete any associated layers
            for _layer in list(self._viewer.layers):
                if _layer.metadata.get("id") == key:
                    self._viewer.layers.remove(_layer)

        # reverse the layers to show the last added layer first
        self._image_combo.clear()
        self._image_combo.addItems(reversed(image_layers))

        # add any newely added layers to the _stored_info
        for layer in image_layers:
            if layer not in self._stored_info:
                self._stored_info[layer] = {
                    AUTO_MASK: [],
                    PREDICTOR: {
                        IMAGE_SET: False,
                        EMBEDDINGS: None,
                        POINTS: {MASKS: [], SCORES: [], COORDS: [], FWD_BKG: []},
                        POINTS_FB: {MASKS: [], SCORES: [], COORDS: [], FWD_BKG: []},
                        BOXES: {MASKS: [], SCORES: [], COORDS: [], FWD_BKG: []},
                    },
                }

    def _on_layer_selected(self, e: Event) -> None:
        """Change the current image to the selected layer and update the combo box."""
        with contextlib.suppress(Exception):
            active_layer = cast("napari.layers", e.source.active)
            layer_name = active_layer.name

            prompt_layer = False
            # in case we select a points or shapes layer
            if _id := active_layer.metadata.get("id"):
                layer_name = _id
                prompt_layer = True

            # set the correct prompt type in the prompt widget combo
            if prompt_layer:
                prompt_type = active_layer.metadata.get("prompt")
                self._predictor_widget.setMode(prompt_type)

            # set the correct image in the clayer combo
            if self._current_image != layer_name:
                self._current_image = layer_name
                self._image_combo.setCurrentText(layer_name)

    # ========================MODEL=========================

    def _on_load(self) -> None:
        """Load the SAM model."""
        self._sam = None
        self._predictor = None

        model_checkpoint, model_type = self._model_group.value()

        self._load_info_lbl.setText("Loading model...")
        logging.info("Loading model...")

        self._model_group.setEnabled(False)

        self._load_worker(model_checkpoint, model_type)

    def _load_worker(self, model_checkpoint: str, model_type: str) -> None:
        self._info_lbl.setText("")
        create_worker(
            self._load,
            model_checkpoint=model_checkpoint,
            model_type=model_type,
            _start_thread=True,
            _connect={
                "yielded": self._update_info,
                "finished": self._on_loaded_finished,
            },
        )

    def _update_info(self, loaded: bool) -> None:
        """Update the info label."""
        if loaded:
            self._stored_info["model"] = {
                "model_checkpoint": self._model_checkpoint,
                "model_type": self._model_type,
            }
            self._enable_widgets(True)
            _loaded_status = f"Model loaded successfully.\nUsing: {self._device}"
            logging.info(_loaded_status)
        else:
            self._enable_widgets(False)
            _loaded_status = "Error while loading model!"
            self._sam = None
            self._predictor = None
            logging.error(_loaded_status)

        self._load_info_lbl.setText(_loaded_status)

    def _on_loaded_finished(self) -> None:
        self._model_group.setEnabled(True)

        if self._console:
            self._console.push({"sam": self._sam, PREDICTOR: self._predictor})

    def _load(
        self, model_checkpoint: str, model_type: str
    ) -> Generator[bool, None, None]:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._model_checkpoint = model_checkpoint
            self._model_type = model_type
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
            self._message_to_log("Load a SAM model first.")
            return

        if (
            not self._viewer.layers
            or not self._current_image
            or self._current_image not in self._viewer.layers
        ):
            self._message_to_log("No image layer selected.")
            return

        self._message_to_log("Generating masks...")

        try:
            self._init_generator()
        except Exception as e:  # be more specific
            self._mask_generator = None
            msg = "Error while initializing the Automatic Mask Generator!"
            self._info_lbl.setText(msg)
            logging.exception(e)

        image = _convert_image(self._viewer.layers[self._current_image])

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
            self._mask_generator = cast(
                "SamAutomaticMaskGenerator", self._mask_generator
            )
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
            self._message_to_log("Automatic Mask Generator finished.")
        else:
            self._message_to_log("Error while running the Automatic Mask Generator!")

    @ensure_main_thread  # type: ignore [misc]
    def _display_labels_auto_segmentation(self, masks: list[np.ndarray]) -> None:
        """Display the masks in a stack."""
        layer_name = self._current_image

        final_mask = self._filter_by_area_and_update_labels(masks)

        name = f"{layer_name}_labels [Automatic]"
        try:
            labels = cast("napari.layers.Labels", self._viewer.layers[name])
            labels.data = final_mask
        except KeyError:
            self._viewer.add_labels(final_mask, name=name, metadata={"id": layer_name})

    def _filter_by_area_and_update_labels(self, masks: list[np.ndarray]) -> np.ndarray:
        """Filter the masks by area and update the data for the labels layer."""
        # filter the masks by area
        filtered_masks = self._filter_mask_by_area(masks)
        if len(filtered_masks) == 0:
            self._message_to_log("No masks found. Adjust the area filter.")
            return

        # update the labels layer with the filtered masks
        filtered_for_labels = np.zeros_like(filtered_masks[0], dtype=np.int32)
        for mask in filtered_masks:
            labeled_mask = measure.label(mask)
            labeled_mask[labeled_mask != 0] += filtered_for_labels.max()
            filtered_for_labels += labeled_mask

        return filtered_for_labels

    def _filter_mask_by_area(self, masks: list[np.ndarray]) -> list[np.ndarray]:
        """Filter the masks by area."""
        min_area, max_area = self._automatic_seg_group.value()
        return [
            mask["segmentation"]
            for mask in masks
            if (mask["area"] >= min_area and mask["area"] <= max_area)
        ]

    def _on_filter(self) -> None:
        """Filter the masks by area and display the labels."""
        labels_name = f"{self._current_image}_labels [Automatic]"
        try:
            labels_layer = cast(
                "napari.layers.Labels", self._viewer.layers[labels_name]
            )
        except KeyError:
            self._message_to_log(
                "No Label Layer found. Run the `Automatic Mask Generator` first."
            )
            return

        masks = self._stored_info.get(self._current_image, {}).get(AUTO_MASK, [])
        if not masks:
            self._message_to_log(
                "No masks found. Run the `Automatic Mask Generator first or adjust the "
                "area filter parameters."
            )
            return

        # filter the masks by area
        filtered_masks = self._filter_by_area_and_update_labels(masks)

        # clear the labels layer
        labels_layer.data = np.zeros_like(labels_layer.data)
        labels_layer.data = filtered_masks

    # ====================SET IMAGE TO PREDICTOR====================

    def _prepare_and_run_predictor(self, layer: napari.layers) -> None:
        """Prepare the predictor and run it on the current image."""
        if self._predictor is None:
            self._message_to_log("Load a SAM model first.")
            return

        # if self._current_image not in self._image_set:
        if self._current_image not in self._stored_info:
            self._message_to_log("No image layer selected.")
            return

        predictor_info = self._stored_info[self._current_image][PREDICTOR]
        image_set = predictor_info.get(IMAGE_SET, False)
        embeddings = predictor_info.get(EMBEDDINGS)

        self._enable_all(False)

        # if the image have never been set to the predictor, set it in another thread
        # and when finished, run the predictor
        if not image_set or embeddings is None:
            self._image_to_predictor_msg.show()
            self._message_to_log("Setting the image to the predictor...")
            create_worker(
                self._image_to_predictor,
                _start_thread=True,
                _connect={"finished": lambda: self._on_image_set_finished(layer)},
            )

        # if the image has been set to the predictor but the embeddings are different
        # set the embeddings to the predictor from the _image_set variable
        elif not torch.allclose(self._predictor.features, embeddings):
            self._message_to_log("Setting the image embeddings to the predictor...")
            self._set_image_embeddings_to_predictor(image_set, embeddings)
            prompts = layer.data
            try:
                self._on_predict(self._predictor_widget.mode(), prompts)
            except Exception as e:  # be more specific
                self._enable_all(True)
                logging.exception(e)

        else:
            prompts = layer.data
            try:
                self._on_predict(self._predictor_widget.mode(), prompts)
            except Exception as e:  # be more specific
                self._enable_all(True)
                logging.exception(e)

    def _image_to_predictor(self) -> None:
        """Set the image to the predictor."""
        try:
            image = _convert_image(self._viewer.layers[self._current_image])
            self._predictor = cast("SamPredictor", self._predictor)
            self._predictor.set_image(image)

            # store the image embeddings
            store = self._stored_info[self._current_image][PREDICTOR]
            store[IMAGE_SET] = True
            store[EMBEDDINGS] = self._predictor.get_image_embedding()
            self._stored_info[self._current_image][PREDICTOR] = store

        except Exception as e:  # be more specific
            self._enable_all(True)
            logging.exception(e)
            return

    def _on_image_set_finished(self, layer: napari.layers) -> None:
        """Enable the widget after setting the image to the predictor."""
        self._enable_all(True)

        self._image_to_predictor_msg.hide()

        predictor_info = self._stored_info[self._current_image][PREDICTOR]
        image_set = predictor_info.get(IMAGE_SET)
        embeddings = predictor_info.get(EMBEDDINGS)
        if image_set and embeddings is not None:
            self._message_to_log("Image set to the predictor.")
            prompts = layer.data
            self._on_predict(self._predictor_widget.mode(), prompts)
        else:
            self._message_to_log("Error while setting the image to the predictor!")

    def _set_image_embeddings_to_predictor(
        self, image_set: bool, embeddings: torch.Tensor
    ) -> bool:
        """Set the image embeddings to the predictor."""
        if self._predictor is None:
            self._message_to_log("Load a SAM model first.")
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

        # get the prompts layer metadata
        layers_meta = [
            (lay.metadata.get("prompt"), lay.metadata.get("id"))
            for lay in self._viewer.layers
        ]
        # add the prompt layer depending on the mode
        if mode in [POINTS, POINTS_FB]:
            self._add_points_layer(layers_meta, mode)
        elif mode == BOXES and (BOXES, self._current_image) not in layers_meta:
            self._add_shapes_layer()

    def _add_points_layer(self, layers_meta: list[tuple[str, str]], mode: str) -> None:
        """Add the points layer for the points prompts."""
        meta = {"id": self._current_image}

        if (mode, self._current_image) in layers_meta:
            return

        meta["prompt"] = mode
        name = f"{self._current_image} [{mode.upper()}]"
        # if there is no points layer, add one
        if meta.get("prompt") and name:
            layer = self._viewer.add_points(
                name=name,
                ndim=2,
                metadata=meta,
                edge_color=GREEN,
                face_color=GREEN,
            )
            layer.mode = "add"
            if mode == POINTS:
                layer.events.data.connect(self._data_changed)
            elif mode == POINTS_FB:
                layer.mouse_drag_callbacks.append(self._change_color_on_mouse_click)

    def _add_shapes_layer(self) -> None:
        """Add the shapes layer for the boxes prompts."""
        layer = self._viewer.add_shapes(
            name=f"{self._current_image} [{BOXES.upper()}]",
            ndim=2,
            metadata={"prompt": BOXES, "id": self._current_image},
            face_color="white",
            edge_color=GREEN,
            edge_width=3,
            opacity=0.4,
            blending="translucent",
        )
        layer.mode = "add_rectangle"
        layer.events.data.connect(self._data_changed)

    def _change_color_on_mouse_click(
        self, points_layer: napari.layers.Points, event: Event
    ) -> None:
        # clear any selected points
        points_layer.selected_data = []
        # set the green color (F) on left click and magenta (B) on right click
        if event.button == Qt.MouseButton.LeftButton.value:
            points_layer.current_face_color = GREEN
            points_layer.current_edge_color = GREEN
        else:
            points_layer.current_face_color = MAGENTA
            points_layer.current_edge_color = MAGENTA

    def _on_predict_signal(self) -> None:
        """Run the predictor with foreground and background points."""
        try:
            layer_name = f"{self._current_image} [{POINTS_FB.upper()}]"
            layer = self._viewer.layers[layer_name]
        except KeyError:
            self._message_to_log(
                "No Foreground and Background points layer found. Click on the 'Add "
                "Points Layers' button first and add foreground and background points."
            )
            return

        self._prepare_and_run_predictor(layer)

    def _data_changed(self, event: Event) -> None:
        """Handle the data change event and run the predictor."""
        layer = cast("napari.layers", event.source)

        # clear the prompt layer if predictor is not set
        if self._predictor is None:
            # block to avoid recursion
            with layer.events.data.blocker():
                layer.data = []
            self._message_to_log("Load a SAM model first.")
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
                    labels = cast("napari.layers.Labels", self._viewer.layers[name])
                    self._viewer.layers.remove(labels)

            else:
                # if the use removed a prompt, but we still have some prompts left
                # we need to re-run the predictor with the new prompts.

                # if there is only one point, we need to clear the stored prompts
                if len(layer.data) == 1:
                    self._clear_mode_info(mode)

                # clearing the labels layer
                with contextlib.suppress(KeyError):
                    labels = cast("napari.layers.Labels", self._viewer.layers[name])
                    labels.data = np.zeros_like(labels.data)
                self._prepare_and_run_predictor(layer)

        elif len(layer.data) > len(coords):
            self._prepare_and_run_predictor(layer)

    def _clear_mode_info(self, mode: str) -> None:
        """Clear the stored info for the current mode."""
        self._stored_info[self._current_image][PREDICTOR][mode][COORDS] = []
        self._stored_info[self._current_image][PREDICTOR][mode][FWD_BKG] = []
        self._stored_info[self._current_image][PREDICTOR][mode][MASKS] = []
        self._stored_info[self._current_image][PREDICTOR][mode][SCORES] = []

    def _on_predict(self, mode: str, prompts: np.ndarray) -> None:
        """Prepare the prompts and run the predictor."""
        if self._sam is None or self._predictor is None:
            self._message_to_log("Load a SAM model first.")
            self._enable_all(True)
            return

        # if shapes
        if isinstance(prompts, list) and not prompts:
            self._message_to_log("No boxes prompts found. Add any boxes first.")
            self._enable_all(True)
            return
        # if points
        elif isinstance(prompts, np.ndarray) and prompts.size == 0:
            self._message_to_log("No points prompts found. Add any points first.")
            self._enable_all(True)
            return

        self._message_to_log(f"Running Predictor with {mode} Prompts...")

        if mode in {POINTS, POINTS_FB}:
            # invert the points to be in the correct (x, y) format
            prompts = list(np.flip(prompts, axis=-1))
        else:  # mode == BOXES:
            # prompt as need to be top_left and bottom_right coordinates
            boxes = []
            for box in prompts:
                box = np.stack([box.min(axis=0), box.max(axis=0)], axis=0)
                box = np.flip(box, -1).reshape(-1)[None, ...]
                boxes.append([box])
            prompts = boxes

        if mode == POINTS_FB:
            updated_prompts = prompts
        else:
            # if the current image has already prompts for the current mode and the
            # length of the new prompts is less than the existing one, means that the
            # user removed a prompts, so we need run the predictor with the new prompts
            predictor_info = self._stored_info[self._current_image][PREDICTOR]
            if stored_prompts := predictor_info.get(mode, []):
                n_prompts = len(stored_prompts[COORDS])
                # use only the last added prompt
                updated_prompts = [prompts[-1]] if len(prompts) > n_prompts else prompts
            else:
                updated_prompts = prompts

        # update the prompts coordinates and forward or background points.
        # NOTE: for the POINTS_FB mode, the forward or background points will be
        # updated in the _predict_with_points_fb method
        self._stored_info[self._current_image][PREDICTOR][mode][COORDS] = prompts
        self._stored_info[self._current_image][PREDICTOR][mode][FWD_BKG] = [
            [1] * len(prompts)
        ]

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
            self._predictor = cast("SamPredictor", self._predictor)

            store = self._stored_info[self._current_image][PREDICTOR][mode]

            if mode == POINTS:
                masks, scores = self._predict_with_points(prompts, store)
            elif mode == POINTS_FB:
                masks, scores = self._predict_with_points_fb(prompts, store)
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
        self._predictor = cast("SamPredictor", self._predictor)
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

    def _predict_with_points_fb(
        self, prompts: np.ndarray, store: dict
    ) -> tuple[list[np.ndarray], list[float]]:
        layer = cast(
            "napari.layers.Points",
            self._viewer.layers[f"{self._current_image} [{POINTS_FB.upper()}]"],
        )
        # get foreground or background points depending on the color
        point_labels = [
            0 if all(layer.face_color[idx] == MAGENTA_CODE) else 1
            for idx, _ in enumerate(prompts)
        ]

        self._predictor = cast("SamPredictor", self._predictor)
        masks, score, _ = self._predictor.predict(
            point_coords=np.array(prompts),
            point_labels=np.array(point_labels),
            multimask_output=False,
        )

        store[MASKS] = masks
        store[SCORES] = score
        store[FWD_BKG] = point_labels

        self._stored_info[self._current_image][PREDICTOR][POINTS_FB] = store

        return masks, score

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
        self._predictor = cast("SamPredictor", self._predictor)
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
            self._message_to_log("Predictor finished.")
        else:
            self._message_to_log("Error while running the Predictor!")

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

            if mode == POINTS_FB:
                try:
                    # get the labels layer and the current data to be updated
                    labels_layer = cast(
                        "napari.layers.Labels", self._viewer.layers[name]
                    )
                    labels = self._update_labels_from_masks(
                        stored_masks, labels_layer.data
                    )
                    labels_layer.data = labels
                except KeyError:
                    labels = self._update_labels_from_masks(stored_masks)
                    self._viewer.add_labels(
                        labels, name=name, metadata={"id": layer_name}
                    )

                # clear the points_fb layer
                with contextlib.suppress(KeyError):
                    layer = self._viewer.layers[f"{layer_name} [{mode.upper()}]"]
                    with layer.events.data.blocker():
                        layer.data = []

            else:
                labels = self._update_labels_from_masks(stored_masks)
                try:
                    labels_layer = cast(
                        "napari.layers.Labels", self._viewer.layers[name]
                    )
                    labels_layer.data = labels
                except KeyError:
                    self._viewer.add_labels(
                        labels, name=name, metadata={"id": layer_name}
                    )

        # keep the prompt layer as the active layer
        prompt_layer = self._viewer.layers[f"{layer_name} [{mode.upper()}]"]
        self._viewer.layers.selection.active = prompt_layer

    def _update_labels_from_masks(
        self, masks: list[np.ndarray], labels_data: np.ndarray | None = None
    ) -> np.ndarray:
        """Create the mask data to be used in the labels layer."""
        labels_data = (
            np.zeros_like(masks[0], dtype=np.int32)
            if labels_data is None
            else labels_data
        )
        for mask in masks:
            labeled_mask = measure.label(mask)
            labeled_mask[labeled_mask != 0] += labels_data.max()
            labels_data += labeled_mask
        return labels_data
