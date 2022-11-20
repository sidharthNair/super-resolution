import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SCALE_FACTOR = 4
HR_SIZE = 96
LR_SIZE = HR_SIZE // SCALE_FACTOR
IN_CHANNELS = 3

