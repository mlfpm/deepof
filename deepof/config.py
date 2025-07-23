from contextvars import ContextVar

suppress_warnings_context = ContextVar('suppress_warnings', default=True)


PROGRESS_BAR_FIXED_WIDTH = 30
ONE_ANIMAL_COLOR_MAP = ['#1f77b4','#17becf', '#9467bd', '#bcbd22', '#d62728', '#8c564b', '#ff7f0e', '#2ca02c', '#7f7f7f'] #9 colors
TWO_ANIMALS_COLOR_MAP = ['#081ee4', '#aa2e47', '#d62246', '#179c79', '#06d6a0', '#0b565f', '#028090', '#c4a31e', '#f1c40f', '#1f77b4','#17becf', '#9467bd', '#bcbd22', '#d62728', '#8c564b', '#ff7f0e', '#2ca02c', '#aec7e8', '#9edae5','#c5b0d5', '#dbdb8d','#ff9896','#c49c94','#ffbb78','#98df8a', '#7f7f7f', '#c7c7c7'] #27 colors
DEEPOF_8_BODYPARTS = ['Center', 'Left_ear', 'Left_fhip', 'Nose', 'Right_ear', 'Right_fhip', 'Tail_base', 'Tail_tip']
DEEPOF_11_BODYPARTS = ['Center', 'Left_bhip', 'Left_ear', 'Left_fhip', 'Nose', 'Right_bhip', 'Right_ear', 'Right_fhip', 'Spine_1', 'Spine_2', 'Tail_base']
DEEPOF_14_BODYPARTS = ['Center', 'Left_bhip', 'Left_ear', 'Left_fhip', 'Nose', 'Right_bhip', 'Right_ear', 'Right_fhip', 'Spine_1', 'Spine_2', 'Tail_1', 'Tail_2', 'Tail_base', 'Tail_tip']
ROI_COLORS = [(204,  20,  20),
       (204, 131,  20),
       (167, 204,  20),
       ( 57, 204,  20),
       ( 20, 204,  94),
       ( 20, 204, 204),
       ( 20,  94, 204),
       ( 57,  20, 204),
       (167,  20, 204),
       (204,  20, 131),
       (153,  15,  15),
       (153,  98,  15),
       (125, 153,  15),
       (43, 153,  15),
       (15, 153,  70),
       (15, 153, 153),
       (15,  70, 153),
       (43,  15, 153),
       (125,  15, 153),
       (153,  15,  98)]
BODYPART_COLORS = [(0,  0,  255),
       (255, 0,  0),
       (0, 255,  0),
       ( 255, 255,  0),
       ( 0, 255, 255),
       ( 255, 0, 255),
       (0,  0,  125),
       ( 125,  0,  0),
       (0,  125,  0),
       ( 125, 125,  0),
       ( 0, 125, 125),
       ( 125, 0, 125),
       (125, 153,  15),
       (43, 153,  15),
       (15, 153,  70),
       (15, 153, 153),
       (15,  70, 153),
       (43,  15, 153),
       (125,  15, 153),
       (153,  15,  98)]
IMG_H_MAX = 700
IMG_W_MAX = 1000