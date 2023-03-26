import torch.nn as nn
import torch.nn.functional as F

# 定义Q网络
class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class DeepQNetwork(nn.Module):
#     def __init__(self):
#         super(DeepQNetwork, self).__init__()
#
#         self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(nn.Linear(64, 1))
#
#         self._create_weights()
#
#     def _create_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#
#         return x
