import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.num_actions=num_actions
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
                    nn.Linear(512, num_actions),
                    nn.Softmax(dim=1)
                )
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward (self, x):
        conv_out = self.conv_stack(x).view(x.size()[0], -1) # batch_size * (C*H*W)
        fc_out = self.fc_stack(conv_out)
        return fc_out
 
    
    def get_action(self, state):
        state = torch.tensor(np.array([state]))
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
