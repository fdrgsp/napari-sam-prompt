from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class AutoMaskGeneratorWidget(QGroupBox):
    """Widget for setting up the parameters for the automatic mask generator.

    When the "Generate Masks" button is clicked, the `generateSignal` is emitted.
    """

    generateSignal = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setTitle("SAM Automatic Mask Generator")

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

        _options_wdg = QWidget()
        _options_wdg_layout = QGridLayout(_options_wdg)

        _options_wdg_layout.addWidget(_points_per_side_lbl, 0, 0)
        _options_wdg_layout.addWidget(self._points_per_side, 0, 1)
        _options_wdg_layout.addWidget(_points_per_batch_lbl, 1, 0)
        _options_wdg_layout.addWidget(self._points_per_batch, 1, 1)
        _options_wdg_layout.addWidget(_pred_iou_thresh_lbl, 2, 0)
        _options_wdg_layout.addWidget(self._pred_iou_thresh, 2, 1)
        _options_wdg_layout.addWidget(_stability_score_thresh_lbl, 3, 0)
        _options_wdg_layout.addWidget(self._stability_score_thresh, 3, 1)
        _options_wdg_layout.addWidget(_stability_score_offset_lbl, 4, 0)
        _options_wdg_layout.addWidget(self._stability_score_offset, 4, 1)
        _options_wdg_layout.addWidget(_box_nms_thresh_lbl, 5, 0)
        _options_wdg_layout.addWidget(self._box_nms_thresh, 5, 1)
        _options_wdg_layout.addWidget(_crop_n_layers_lbl, 6, 0)
        _options_wdg_layout.addWidget(self._crop_n_layers, 6, 1)
        _options_wdg_layout.addWidget(_crop_nms_thresh_lbl, 7, 0)
        _options_wdg_layout.addWidget(self._crop_nms_thresh, 7, 1)
        _options_wdg_layout.addWidget(_crop_overlap_ratio_lbl, 8, 0)
        _options_wdg_layout.addWidget(self._crop_overlap_ratio, 8, 1)
        _options_wdg_layout.addWidget(_crop_n_points_downscale_factor_lbl, 9, 0)
        _options_wdg_layout.addWidget(self._crop_n_points_downscale_factor, 9, 1)
        _options_wdg_layout.addWidget(_min_mask_region_area_lbl, 10, 0)
        _options_wdg_layout.addWidget(self._min_mask_region_area, 10, 1)
        _options_wdg_layout.addWidget(_output_mode_lbl, 11, 0)
        _options_wdg_layout.addWidget(self._output_mode, 11, 1)
        _options_wdg_layout.addWidget(_min_area_lbl, 12, 0)
        _options_wdg_layout.addWidget(self._min_area, 12, 1)
        _options_wdg_layout.addWidget(_max_area_lbl, 13, 0)
        _options_wdg_layout.addWidget(self._max_area, 13, 1)

        self._generate_mask_btn = QPushButton("Generate Masks")
        self._generate_mask_btn.clicked.connect(self.generateSignal.emit)

        collapsible = QCollapsible("Options")
        collapsible.layout().setContentsMargins(0, 0, 0, 0)

        collapsible.addWidget(_options_wdg)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)
        main_layout.addWidget(collapsible)
        main_layout.addWidget(self._generate_mask_btn)
