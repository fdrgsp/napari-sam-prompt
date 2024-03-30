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
STANDARD = "standard"
LOOP = "loop"


class PredictorWidget(QGroupBox):
    addLayersSignal = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("SAM Predictor")

        self._mode_combo = QComboBox()
        self._mode_combo.addItems([POINTS, BOXES])
        self._mode_combo.currentIndexChanged.connect(self._on_combo_changed)

        self._add_layer_btn = QPushButton()
        self._add_layer_btn.clicked.connect(self._on_add_layers)

        # main layout
        _layout = QHBoxLayout(self)
        _layout.setSpacing(5)
        _layout.setContentsMargins(10, 10, 10, 10)

        _layout.addWidget(self._mode_combo)
        _layout.addWidget(self._add_layer_btn)

        self._on_combo_changed()

    def mode(self) -> str:
        return str(self._mode_combo.currentText())

    def setMode(self, mode: str) -> None:
        self._mode_combo.setCurrentText(mode)

    def _on_add_layers(self) -> None:
        """Emit the `addLayersSignal` with the current widget value."""
        self.addLayersSignal.emit(self.mode())

    def _on_combo_changed(self) -> None:
        self._add_layer_btn.setText(f"Add {self.mode().split(' ')[0]} Layers")
