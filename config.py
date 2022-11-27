import torch

PRETRAIN = True
TEST = True
LOAD = True
SAVE = True
GEN_FILE = 'gen.tar'
DISC_FILE = 'disc.tar'
LOSSES_FILE = 'losses.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 4
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
SCALE_FACTOR = 4
HR_SIZE = 96
LR_SIZE = HR_SIZE // SCALE_FACTOR
CHANNELS = 3
