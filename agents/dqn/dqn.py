import numpy as np
import torch
import torch.nn as nn


class DQNNet (nn.Module):
    """
    simple DQN net implementation
    -----------------------------------------------------    
    Parameters:
    input_shape: shape of observation
    n_actions: size of action space
    """

    def __init__ (self, input_shape, n_actions):
        super(DQNNet, self).__init__()
        in_channels = input_shape[0]
        self.conv_stack = nn.Sequential(
                            nn.Conv2d(in_channels, 32, kernel_size= 8, stride= 4),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=4, stride= 2),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride= 1),
                            nn.ReLU()
                        )
        
        # get convolution output size
        sample_in = torch.zeros(1, *input_shape) # 1 * C * H * W
        conv_out_shape = self.conv_stack(sample_in).size()
        conv_out_flattened = np.prod(conv_out_shape)

    
        # fully connected layer    
        self.fc_stack = nn.Sequential(
                    nn.Linear(conv_out_flattened, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_actions)
                )

    def forward (self, x):
        conv_out = self.conv_stack(x).view(x.size()[0], -1) # batch_size * (C*H*W)
        fc_out = self.fc_stack(conv_out)
        return fc_out
