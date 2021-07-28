import logging
import sys

from os import path

# Basic logging configuration
LOGGING_CONFIG = {
    "stream": sys.stdout,
    "format": "%(levelname)s:%(asctime)s:%(message)s",
    "level": logging.INFO,
    "datefmt": "%Y-%m-%d %H:%M:%S"}

# Delta tolerance to the original weat test results
WEAT_TEST_TOLERANCE = 0.03

# Delta tolerance to the original rnsb test results
RNSB_TEST_TOLERANCE = 0.0001

# Directory containing the pre-trained word vector files
WORD_VECTOR_DIR = path.join("word_vectors")

# Lower and upper score limits (in that order) of different bias metrics
BIAS_METRIC_LIMITS = {
    "ect": (-1, 1),
    "rnsb": (0, 1),
    "weat": (-2, 2)}

# "No bias" scores of different bias metrics
BIAS_METRIC_ZERO = {
    "ect": 0,
    "rnsb": 0,
    "weat": 0}

# Color shades (RGBA format)
COLORS = {
    "blue": {
        "dark": (0, 0, 1 / 255 * 115, 1.0),
        "medium": (0, 0, 1.0, 0.3),
        "light": (0, 0, 1.0, 0.25)},
    "red": {
        "dark": (1 / 255 * 139, 1 / 255 * 20, 0, 1.0),
        "medium": (1.0, 1 / 255 * 20, 0, 0.25),
        "light": (1.0, 1 / 255 * 20, 0, 0.1)}}
