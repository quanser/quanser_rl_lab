# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Launches Isaac Sim once for the whole test session.

This has to live here rather than in the test modules. Isaac Sim can only be
launched once per process, and pytest imports every test module into the same
process during collection -- so a second module calling AppLauncher hangs the
run at "collecting". conftest.py is imported before any test module, which
makes it the one place the app can be started exactly once.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app
