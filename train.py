import numpy as np
import torch
import config
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import ContentLoss
from models import Generator, Discriminator
from data import CustomDataset
from test import test_generator

torch.backends.cudnn.benchmark = True

def train_gan(loader, disc, gen, disc_optimizer, gen_optimizer, pretrain):
    bce = nn.BCEWithLogitsLoss()
    content_loss = ContentLoss()
    mse = mse = nn.MSELoss()
    tqdm_loader = tqdm(loader, leave=True)

    for lr, hr in tqdm_loader:
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
        if pretrain:
            # In the SRGAN paper, they use a pre-trained version of SRResNet as
            # their starting point to avoid converging to undesired local optima
            # This pretraining step serves to accomplish the same.
            gen_loss = mse(fake, hr)
        else:
            disc_fake = disc(fake)
            c_loss = 0.006 * content_loss(fake, hr)
            a_loss = 0.001 * bce(disc_fake, torch.ones_like(disc_fake))
            gen_loss = c_loss + a_loss

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

    return gen_loss.cpu().detach().numpy(), disc_loss.cpu().detach().numpy()

def save(filename, model, optimizer, losses):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses": losses,
    }
    torch.save(checkpoint, filename)


def load(filename, model, optimizer, losses, lr):
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    losses.extend(checkpoint["losses"])
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

    gen_losses = []
    disc_losses = []

    if config.LOAD:
        print('LOADING MODELS FROM CHECKPOINT FILES')
        load(config.GEN_FILE, gen, gen_optimizer, gen_losses, config.LEARNING_RATE)
        load(config.DISC_FILE, disc, disc_optimizer, disc_losses, config.LEARNING_RATE)

    gen.train()
    disc.train()

    base = len(gen_losses) - 1
    for epoch in range(config.EPOCHS):
        print(f'EPOCH: {base + epoch}')
        gen_loss, disc_loss = train_gan(loader, disc, gen, disc_optimizer, gen_optimizer, config.PRETRAIN)
        print(f'Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)

        if config.SAVE:
            save(config.GEN_FILE, gen, gen_optimizer, gen_losses)
            save(config.DISC_FILE, disc, disc_optimizer, disc_losses)
            np.savetxt(config.LOSSES_FILE, np.column_stack((gen_losses, disc_losses)), delimiter=',', fmt='%s', header='gen_loss, disc_loss')

        if config.TEST:
            test_generator(gen, prefix=f'{base + epoch}_')

if __name__ == "__main__":
    main()