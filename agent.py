# Defines the class Agent that brings the sacacde and fovea networks together, takes care of getting the patch, etc...
import torch as tch
from networks import ConvolutionalRecurrentNetwork


class SaccadeAgent(tch.nn.Module):
    def __init__(self, peripheral_net_params, foveal_net_params) -> None:
        super().__init__()
        self.device = tch.device('cuda')
        self.peripheral_net = ConvolutionalRecurrentNetwork(peripheral_net_params)
        self.foveal_net = ConvolutionalRecurrentNetwork(foveal_net_params)
        self.to(self.device)

    def forward(self, peripheral_image, foveal_image):
        return self.peripheral_net(peripheral_image), self.foveal_net(foveal_image)
    
    def get_saccade(self, peripheral_image):
        return self.peripheral_net(peripheral_image)
    
    def get_homing(self, foveal_image):
        return self.foveal_net(foveal_image)
    
    