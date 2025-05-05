from __future__ import annotations

from typing import Iterable

__all__ = ["TauHelpFormatter"]

import textwrap
from argparse import (
    Action,
    ArgumentDefaultsHelpFormatter,
    RawDescriptionHelpFormatter,
    _MutuallyExclusiveGroup,
)


class TauHelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """Custom help formatter class to override standard formatting output."""

    def __init__(self, prog):
        super().__init__(prog, width=79, max_help_position=6)

    def _format_action_invocation(self, action: Action) -> str:
        """Modify help message format to remove duplicate metavar entries."""
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join(action.option_strings) + " " + args_string

    def _split_lines(self, text: str, width: int) -> list[str]:
        """Split text into lines; add a line break after each option block."""
        return super()._split_lines(text, width) + [""]
