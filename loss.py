import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE

# Implementation of content loss for SRGAN as defined in https://arxiv.org/pdf/1609.04802.pdf


class ContentLoss(nn.Module):
    # Instead of relying on pixel-wise MSE loss for content loss, the designers
    # of SRGAN defined a new term called VGG loss. This is obtained by taking
    # the euclidean distance between the feature representation of the reference image and
    # the reconstructed image (which is found by taking the feature map obtained in
    # the j-th convolution, after activation and before the i-th maxpooling layer, of
    # the pre-trained VGG19 network).
    def __init__(self):
        super().__init__()
        # Here we take the output up to the 18th layer, which is the activation before
        # the 3rd maxpooling layer in VGG19. Other options we could use are: 4, 9, 27, 36.
        self.vgg19 = models.vgg19(pretrained=True).features[:18].eval().to(DEVICE)
        # Euclidean distance is just MSE loss between feature map and reference
        self.dist = nn.MSELoss()

        # We don't want to modify the pre-trained vgg19 network, so freeze its parameters
        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, reconstructed, reference):
        # Extract the feature maps from the images
        reconstructed_features = self.vgg19(reconstructed)
        reference_features = self.vgg19(reference)

        # Loss is the euclidean distance between the feature maps
        loss = self.dist(reconstructed_features, reference_features)
        return loss
