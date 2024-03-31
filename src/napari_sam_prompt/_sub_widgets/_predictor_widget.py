from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
POINTS = "Points Prompts"
BOXES = "Boxes Prompts"
POINTS_FB = "Points Prompts [F&B]"


class PredictorWidget(QGroupBox):
    addLayersSignal = Signal(object)
    predictSignal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("SAM Predictor")

        self._mode_combo = QComboBox()
        self._mode_combo.addItems([POINTS, POINTS_FB, BOXES])
        self._mode_combo.currentIndexChanged.connect(self._on_combo_changed)

        self._add_layer_btn = QPushButton()
        self._add_layer_btn.clicked.connect(self._on_add_layers)

        self._predict_btn = QPushButton("Predict")
        self._predict_btn.setToolTip(
            "'Left-Click' to add foreground points and 'Right-Click' to add "
            "background points. Press 'Predict' to predict the labels."
        )
        self._predict_btn.clicked.connect(self.predictSignal.emit)

        # main layout
        _layout = QHBoxLayout(self)
        _layout.setSpacing(5)
        _layout.setContentsMargins(10, 10, 10, 10)

        _layout.addWidget(self._mode_combo)
        _layout.addWidget(self._add_layer_btn)
        _layout.addWidget(self._predict_btn)

        self._on_combo_changed()

    def mode(self) -> str:
        return str(self._mode_combo.currentText())

    def setMode(self, mode: str) -> None:
        self._mode_combo.setCurrentText(mode)

    def _on_add_layers(self) -> None:
        """Emit the `addLayersSignal` with the current widget value."""
        self.addLayersSignal.emit(self.mode())

    def _on_combo_changed(self) -> None:
        mode = self.mode()
        self._predict_btn.show() if mode == POINTS_FB else self._predict_btn.hide()
        self._add_layer_btn.setText(f"Add {self.mode().split(' ')[0]} Layers")
