"""
Default experiment configuration.

These values reproduce the defaults used in the original notebook.
Runtime paths such as DEAP_ROOT and IMAGE_ROOT are intentionally
not defined here because they depend on the execution environment.
"""

# Experiment
LABEL_TYPE = 1       # 0 = valence, 1 = arousal
MAX_SUBJECT_ID = 22
SEED = 42

# Video
NUM_FRAMES = 18
FREEZE_BACKBONE = True

# Training
BATCH_SIZE = 4
WEIGHT_DECAY = 1e-5
EPOCHS = 20
LR = 1e-3
TRAIN_RATIO = 0.75
NUM_WORKERS = 4