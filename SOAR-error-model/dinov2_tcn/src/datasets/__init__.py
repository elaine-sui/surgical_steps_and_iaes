from .transforms import *

from .data import (
    DINObTemporalCollator,
    DINObTemporalWithPNRCollator,
    DINOCollator,
)

ALL_TEMPORAL_COLLATORS_WITH_PNR = {
    "DINOv2": DINObTemporalWithPNRCollator,
}

ALL_TEMPORAL_COLLATORS = {
    "DINOv2": DINObTemporalCollator,
}

ALL_COLLATORS = {
    'DINOv2': DINOCollator,
    'DINOv2-base-2head': DINOCollator,
    'DINOv2-base-multiclass': DINOCollator
}