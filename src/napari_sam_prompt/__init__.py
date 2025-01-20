"""A napari plugin that implements SAM prompts predictor."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari-sam-prompt")
except PackageNotFoundError:
    __version__ = "uninstalled"

from ._sam_prompt_widget import SamPromptWidget

__all__ = ["SamPromptWidget", "__version__"]
