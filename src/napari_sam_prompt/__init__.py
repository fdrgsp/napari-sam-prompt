"""A napari plugin that implements SAM prompts predictor."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._sam_prompt_widget import SamPromptWidget

__all__ = ["__version__", "SamPromptWidget"]
