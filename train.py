import torch
import config
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import ContentLoss
from models import Generator, Discriminator
from data import CustomDataset

torch.backends.cudnn.benchmark = True

# In the SRGAN paper, they use a pre-trained version of SRResNet as
# their starting point to avoid converging to undesired local optima
# This pretraining step serves to accomplish the same.
def pretrain_generator(loader, gen, optimizer):
    mse = mse = nn.MSELoss()
    loop = tqdm(loader, leave=True)

    for lr, hr in loop:
        lr = lr.to(config.DEVICE)
        hr = hr.to(config.DEVICE)

        fake = gen(lr)
        loss = mse(fake, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_gan(loader, disc, gen, disc_optimizer, gen_optimizer):
    bce = nn.BCEWithLogitsLoss()
    content_loss = ContentLoss()
    loop = tqdm(loader, leave=True)

    for lr, hr in loop:
        lr = lr.to(config.DEVICE)
        hr = hr.to(config.DEVICE)

        # Train Discriminator
        fake = gen(lr)
        disc_real = disc(hr)
        disc_fake = disc(fake.detach())
        disc_loss = bce(disc_real, torch.ones_like(disc_real)) + bce(disc_fake, torch.zeros_like(disc_fake))

        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # Train Generator
        disc_fake = disc(fake)
        gen_loss = 0.006 * content_loss(fake, hr) + 0.001 * bce(disc_fake, torch.ones_like(disc_fake))

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

def save(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load(filename, model, optimizer, lr):
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main():
    dataset = CustomDataset("data/")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.WORKERS,
    )
    gen = Generator(in_channels=config.CHANNELS, scaling_factor=config.SCALE_FACTOR).to(config.DEVICE)
    disc = Discriminator(in_channels=config.CHANNELS).to(config.DEVICE)
    gen_optimizer = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE)
    disc_optimizer = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE)

    if config.LOAD:
        load(config.CHECKPOINT_GEN, gen, gen_optimizer, config.LEARNING_RATE)
        load(config.CHECKPOINT_DISC, disc, disc_optimizer, config.LEARNING_RATE)


    for epoch in range(config.EPOCHS):
        if config.PRETRAIN:
            pretrain_generator(loader, gen, gen_optimizer)
        else:
            train_gan(loader, disc, gen, disc_optimizer, gen_optimizer)

        if config.SAVE:
            save(gen, gen_optimizer, config.CHECKPOINT_GEN)
            save(disc, disc_optimizer, config.CHECKPOINT_DISC)

if __name__ == "__main__":
    main()