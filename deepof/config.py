from contextvars import ContextVar
import numpy as np
from enum import Enum

suppress_warnings_context = ContextVar('suppress_warnings', default=True)


PROGRESS_BAR_FIXED_WIDTH = 30
ONE_ANIMAL_COLOR_MAP = ['#1f77b4','#17becf', '#9467bd', '#bcbd22', '#d62728', '#8c564b', '#ff7f0e', '#2ca02c', '#7f7f7f'] #9 colors
TWO_ANIMALS_COLOR_MAP = ['#081ee4', '#aa2e47', '#d62246', '#179c79', '#06d6a0', '#0b565f', '#028090', '#c4a31e', '#f1c40f', '#1f77b4','#17becf', '#9467bd', '#bcbd22', '#d62728', '#8c564b', '#ff7f0e', '#2ca02c', '#aec7e8', '#9edae5','#c5b0d5', '#dbdb8d','#ff9896','#c49c94','#ffbb78','#98df8a', '#7f7f7f', '#c7c7c7'] #27 colors
DEEPOF_8_BODYPARTS = ['Center', 'Left_ear', 'Left_fhip', 'Nose', 'Right_ear', 'Right_fhip', 'Tail_base', 'Tail_tip']
DEEPOF_11_BODYPARTS = ['Center', 'Left_bhip', 'Left_ear', 'Left_fhip', 'Nose', 'Right_bhip', 'Right_ear', 'Right_fhip', 'Spine_1', 'Spine_2', 'Tail_base']
DEEPOF_14_BODYPARTS = ['Center', 'Left_bhip', 'Left_ear', 'Left_fhip', 'Nose', 'Right_bhip', 'Right_ear', 'Right_fhip', 'Spine_1', 'Spine_2', 'Tail_1', 'Tail_2', 'Tail_base', 'Tail_tip']
SINGLE_BEHAVIORS=["climb-arena", "sniff-arena", "immobility", "stat-lookaround", "stat-active", "stat-passive", "moving", "sniffing", "missing"]
SYMMETRIC_BEHAVIORS=["nose2nose","sidebyside","sidereside"]
ASYMMETRIC_BEHAVIORS=["nose2tail","nose2body","following"]
CONTINUOUS_BEHAVIORS=["distance","cum-distance","speed"]
CONTINUOUS_UNITS=["[mm]", "[mm]", "[mm/s]"]

ROI_COLORS = [(204, 20, 20),
       (204, 131, 20),
       (167, 204, 20),
       (57, 204, 20),
       (20, 204, 94),
       (20, 204, 204),
       (20, 94, 204),
       (57, 20, 204),
       (167, 20, 204),
       (204, 20, 131),
       (153, 15, 15),
       (153, 98, 15),
       (125, 153, 15),
       (43, 153, 15),
       (15, 153, 70),
       (15, 153, 153),
       (15, 70, 153),
       (43, 15, 153),
       (125, 15, 153),
       (153, 15, 98)]
ARENA_COLOR = (40, 86, 236)

BODYPART_COLORS = [(0, 0, 255),
       (255, 0, 0),
       (0, 255, 0),
       (255, 255, 0),
       (0, 255, 255),
       (255, 0, 255),
       (0, 0, 125),
       (125, 0, 0),
       (0, 125, 0),
       (125, 125, 0),
       (0, 125, 125),
       (125, 0, 125),
       (125, 153, 15),
       (43, 153, 15),
       (15, 153, 70),
       (15, 153, 153),
       (15, 70, 153),
       (43, 15, 153),
       (125, 15, 153),
       (153, 15, 98)]
IMG_H_MAX = 700
IMG_W_MAX = 1000

# ENUMS # 

# DeepOF saves all distances internally in mm, correspondingly thsi enum contains appropriate conversion factors
class DistanceUnit(Enum):
    pixel = 0.0 
    px = 0.0
    mm = 1.0 # identity, measures are saved in mm per default
    millimeter = 1.0
    cm = 10
    centimeter = 10
    m = 1000
    meter = 1000
    km = 1000000
    kilometer = 1000000
    inch = 25.4
    foot = 304.8
    yard = 914.4
    mile = 1609000


    def factor(self, mm_to_pix = None):
        """Multiplier to convert mm -> this unit. mm_to_pix can be scalar or array-like."""
        if self in (DistanceUnit.px, DistanceUnit.pixel):
            if mm_to_pix is None:
                raise ValueError('For pixel conversions a mm_to_pix conversion factor must be given!')
            return np.asarray(mm_to_pix, dtype=float)
        return 1.0 / self.value

    @classmethod
    def parse(cls, unit: str) -> "DistanceUnit":
        try:
            return cls[unit]
        except KeyError as e:  # pragma: no cover
            opts = ", ".join(cls.__members__.keys())
            raise ValueError(f'Unknown distance unit "{unit}". Valid options are: {opts}') from e

# DeepOF saves all distances internally in frames, correspondingly this enum calculates appropriate conversion factors
class TimeUnit(Enum):
    fr = 0.0 # identity (frames -> frames)
    frames = 0.0   
    s     = 1.0   # seconds per unit
    seconds = 1.0
    min   = 60.0
    minutes = 60.0
    h     = 3600.0
    hours = 3600.0

    def factor(self, fps: float) -> float:
        """Multiplier to convert frames -> this unit."""
        if self is TimeUnit.frames or fps is None:
            return 1.0
        return 1.0 / (fps * self.value)
    
    @classmethod
    def parse(cls, unit: str) -> "TimeUnit":
        try:
            return cls[unit]
        except KeyError as e:  # pragma: no cover
            opts = ", ".join(cls.__members__.keys())
            raise ValueError(f'Unknown time unit "{unit}". Valid options are: {opts}') from e


# Native time unit in DeepOF is 
class Speed_Unit(Enum):
    mm_s = 1 
    m_s = 0.001
    m_h = 3.6
