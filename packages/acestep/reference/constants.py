"""Immutable identities and format names used by reference capture."""

from __future__ import annotations


ACE_SOURCE_REPOSITORY = "https://github.com/ace-step/ACE-Step-1.5.git"
ACE_SOURCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
ACE_MODEL_ID = "ACE-Step/Ace-Step1.5"
ACE_MODEL_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
PLANNER_MODEL_ID = "ACE-Step/acestep-5Hz-lm-0.6B"
PLANNER_MODEL_REVISION = "148d8ea0225bdab342ee1ae3a354275ccd60ca80"

REFERENCE_TOOL_VERSION = 2
CAPTURE_SCHEMA_VERSION = 2
INPUT_SCHEMA_VERSION = 1
INPUT_ALGORITHM_ID = "ace-seed-v1"
REFERENCE_BACKEND = "pytorch-eager"
REFERENCE_ATTENTION = "eager"
REFERENCE_DCW_BACKEND = "pytorch-wavelets-haar"

HEX_DIGITS = frozenset("0123456789abcdef")
