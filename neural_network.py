from torch import nn
import torch
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_layer=512):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_layer)
        self.layer2 = nn.Linear(hidden_layer, hidden_layer)
        self.layer3 = nn.Linear(hidden_layer, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)