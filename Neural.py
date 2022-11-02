import torch, os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self, input_layer:tuple, output_layer:tuple, hidden_layers:tuple):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = self.get_relu(input_layer, output_layer, hidden_layers)


    def forward_prop(self):
        pass

    def back_prop(self):
        pass

    def get_relu(self, input_layer, output_layer, hidden_layers, layer_type='linear'):
        layers = OrderedDict()
        all_layers = np.vstack((input_layer, output_layer, hidden_layers))


        for i, layer in enumerate(all_layers):
            if layer_type == 'linear':
                layers[f'conv_{i}'] = nn.Linear(layer[0], layer[1])

            if i != (all_layers.shape[0] - 1):
                layers[f'relu_{i}'] = nn.ReLU()

        return nn.Sequential(layers)

a = NeuralNetwork((1000, 512), (512, 512), (512, 10))
print('b')

