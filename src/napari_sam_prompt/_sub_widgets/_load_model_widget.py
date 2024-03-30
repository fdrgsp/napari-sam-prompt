from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class LoadModelWidget(QGroupBox):
    """Widget for loading a model checkpoint."""

    loadSignal = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        model_checkpoint: str = "",
        model_type: str = "",
    ):
        super().__init__(parent)

        self.setTitle("SAM Model Checkpoint")

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
        self._load_module_btn.clicked.connect(self.loadSignal)

        _model_group_layout = QGridLayout(self)
        _model_group_layout.setSpacing(10)
        _model_group_layout.setContentsMargins(10, 10, 10, 10)

        _model_group_layout.addWidget(_model_lbl, 0, 0)
        _model_group_layout.addWidget(self._model_le, 0, 1)
        _model_group_layout.addWidget(self._model_browse_btn, 0, 2)
        _model_group_layout.addWidget(_model_type_lbl, 1, 0)
        _model_group_layout.addWidget(self._model_type_le, 1, 1, 1, 2)
        _model_group_layout.addWidget(self._load_module_btn, 2, 0, 1, 3)

    def _browse_model(self) -> None:
        """Open a file dialog to select the SAM Model Checkpoint."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select the SAM Model Checkpoint to use.", "", "pth(*.pth)"
        )
        if filename:
            self._model_le.setText(filename)

    def value(self) -> tuple[str, str]:
        """Return the model path and model type."""
        return self._model_le.text(), self._model_type_le.text()
