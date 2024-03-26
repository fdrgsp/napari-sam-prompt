from __future__ import annotations

from typing import cast

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
POINTS = "points"
BOXES = "boxes"
STANDARD = "standard"
LOOP = "loop"


class PromptWidget(QWidget):
    """Widget for selecting the predictor type and prompt."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        predictor_prompt: str,  # POINTS or BOXES
    ) -> None:
        super().__init__(parent)

        if predictor_prompt not in (POINTS, BOXES):
            raise ValueError(
                f"Invalid value for `predictor_prompt`: {predictor_prompt}. "
                "It must be either 'points' or 'boxes'."
            )

        self._predictor_prompt = predictor_prompt
        name = self._predictor_prompt.capitalize()

        _predictor_layout = QGridLayout(self)
        _predictor_layout.setSpacing(5)
        _predictor_layout.setContentsMargins(5, 5, 5, 5)
        self._standard_radio = QRadioButton("Standard Predictor")
        self._standard_radio.setChecked(True)
        self._loop_radio = QRadioButton(f"Loop Single {name} Predictor")
        self._add_layer_btn = QPushButton(f"Add {name} Layers")
        self._predict_btn = QPushButton(f"Predict with {name} Prompt")
        _predictor_layout.addWidget(self._standard_radio, 0, 0)
        _predictor_layout.addWidget(self._loop_radio, 0, 1)
        _predictor_layout.addWidget(self._add_layer_btn, 1, 0, 1, 3)
        _predictor_layout.addWidget(self._predict_btn, 2, 0, 1, 3)

    def value(self) -> dict:
        return {
            "predictor_prompt": self._predictor_prompt,
            "predictor_type": (STANDARD if self._standard_radio.isChecked() else LOOP),
        }


class PredictorWidget(QGroupBox):
    predictSignal = Signal(object)
    addLayersSignal = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("SAM Predictor")

        # points predictor
        self._points_predictor = PromptWidget(predictor_prompt=POINTS)
        self._points_predictor._add_layer_btn.clicked.connect(self._on_add_layers)
        self._points_predictor._predict_btn.clicked.connect(self._on_predict)

        # boxes predictor
        self._boxes_predictor = PromptWidget(predictor_prompt=BOXES)
        self._boxes_predictor._loop_radio.setChecked(True)
        self._boxes_predictor._standard_radio.hide()
        self._boxes_predictor._add_layer_btn.clicked.connect(self._on_add_layers)
        self._boxes_predictor._predict_btn.clicked.connect(self._on_predict)

        # tab widget
        self._tab = QTabWidget()
        self._tab.addTab(self._points_predictor, "Points Predictor")
        self._tab.addTab(self._boxes_predictor, "Boxes Predictor")

        # main layout
        _layout = QVBoxLayout(self)
        _layout.setContentsMargins(10, 10, 10, 10)
        _layout.addWidget(self._tab)

    def value(self) -> dict:
        # get the current tab
        current_tab = self._tab.currentIndex()
        # get the current widget in the tab and return its value
        widget = self._tab.widget(current_tab)
        return cast(dict, widget.value())

    def _on_predict(self) -> None:
        """Emit the `predictSignal` with the current widget value."""
        self.predictSignal.emit(self.value())

    def _on_add_layers(self) -> None:
        """Emit the `addLayersSignal` with the current widget value."""
        self.addLayersSignal.emit(self.value())
