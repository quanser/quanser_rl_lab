# Copyright 2026 Quanser Consulting Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateful models of actuator behaviour, stepped inside the physics loop.

Anything here carries per-env state across substeps and must be reset when an
environment resets, which is what separates it from a stateless helper.
"""
