"""Run napari-sam-prompt as a script with `python -m napari_sam_prompt`."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import napari

from napari_sam_prompt._sam_prompt_widget import SamPromptWidget


def main(args: Sequence[str] | None = None) -> None:
    """Run napari-sam-prompt."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Enter string")
    parser.add_argument(
        "-mc",
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint.",
        nargs="?",
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        default=None,
        help="Model checkpoint type.",
        nargs="?",
    )
    parsed_args = parser.parse_args(args)

    viewer = napari.Viewer()
    win = SamPromptWidget(
        viewer=viewer,
        model_checkpoint=parsed_args.model_checkpoint,
        model_type=parsed_args.model_type,
    )
    dw = viewer.window.add_dock_widget(win, name="sam-prompt", area="right")
    if hasattr(dw, "_close_btn"):
        dw._close_btn = False
    napari.run()


if __name__ == "__main__":
    main()
