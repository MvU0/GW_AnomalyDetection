# Configuration of training for anomaly detection model discussed in thesis: 
            Learning Auxiliary Channel Relationships for Glitch Detection in Gravitational Wave Detectors Using Graph Neural Networks

# Experiment parameters
DATASET = GW
SUBSYSTEMS = IMC
BATCH = 129
EPOCHS = 400
SLIDE_WIN = 100
SLIDE_STRIDE = 5
DIM = 128
TOPK = 30
OUT_LAYER_NUM = 1
OUT_LAYER_INTER_DIM = 128
VAL_RATIO = 0.2
TEST_RATIO = 0.1
DECAY = 1e-5
LR = 1e-4
REPORT = val