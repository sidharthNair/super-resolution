import torch

PRETRAIN = True
LOAD = True
SAVE = True
CHECKPOINT_GEN = 'checkpoint_gen.pth'
CHECKPOINT_DISC = 'checkpoint_disc.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 4
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
SCALE_FACTOR = 4
HR_SIZE = 96
LR_SIZE = HR_SIZE // SCALE_FACTOR
CHANNELS = 3
