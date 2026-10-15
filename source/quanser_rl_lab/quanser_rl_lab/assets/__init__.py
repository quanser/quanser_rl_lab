# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot asset configurations and the path to their description files."""

import os

# Convenience paths to the extension root and its sibling data/ directory
QUANSER_RL_LAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory (source/quanser_rl_lab)."""

QUANSER_RL_LAB_ASSETS_DATA_DIR = os.path.join(QUANSER_RL_LAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory, holding robot USD files."""
