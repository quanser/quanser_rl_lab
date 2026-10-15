# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'quanser_rl_lab' python package."""

import os

import toml
from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Only what `quanser_rl_lab` itself imports and pip can actually resolve.
#
# isaaclab, isaaclab_rl and isaaclab_tasks are deliberately absent: they are
# not on PyPI, they are installed from the Isaac Lab source tree, and listing
# them here makes pip fail trying to fetch them. They are declared in
# config/extension.toml instead, which is the Omniverse mechanism for it.
# omni and pxr come from Isaac Sim the same way.
INSTALL_REQUIRES = [
    # torch is intentionally unpinned. Isaac Sim supplies a specific CUDA
    # build (2.7.0+cu128 at time of writing) and a version bound here risks
    # pip replacing it with a wheel that does not match the driver.
    "torch",
    "numpy",
    "gymnasium",
    # The actor and distribution are loaded by import path from the PPO cfg,
    # and both subclass rsl_rl types, so its model API is a hard dependency.
    # Bounded below because RslRlMLPModelCfg's distribution_cfg/class_name
    # form is required, and above because play.py already carries
    # handle_deprecated_rsl_rl_checkpoint for an earlier format change.
    "rsl-rl-lib>=5.0.1,<6",
]

# Installation operation
setup(
    name="quanser_rl_lab",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    author_email=EXTENSION_TOML_DATA["package"]["author_email"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    maintainer_email=EXTENSION_TOML_DATA["package"]["author_email"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        # Not a registered PyPI classifier -- an Isaac Lab convention. Strip
        # the "Isaac Sim ::" lines before any upload to PyPI, which rejects
        # unknown classifiers.
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
