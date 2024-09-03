import torch.nn as nn
import torchvision.models as models

import json

class ResNet50Dropout(nn.Module):
    def __init__(self, weights=None, dropout_rate=0.):
        super(ResNet50Dropout, self).__init__()

        self.weights = weights
        self.model = models.resnet50(weights=self.weights)
        self.dropout_rate = dropout_rate

        self.dropout_positions = []
        if self.dropout_rate > 0:
            self.dropout_positions = self.get_dropout_positions()

        self._add_dropout()

    def get_dropout_positions(self):
        dropout_positions_path = "Domain-Shift-Computer-Vision/utility/data/dropout_positions.json"
        with open(dropout_positions_path, 'r') as json_file:
            dropout_positions = json.load(json_file)
        dropout_positions = dropout_positions["dropout_positions"]

        return dropout_positions

    def _add_dropout(self):
        if 'conv1' in self.dropout_positions:
            self.model.conv1 = nn.Sequential(
                self.model.conv1,
                nn.Dropout(p=self.dropout_rate)
            )

        if 'layer1' in self.dropout_positions:
            self.model.layer1 = nn.Sequential(
                self.model.layer1,
                nn.Dropout(p=self.dropout_rate)
            )

        if 'layer2' in self.dropout_positions:
            self.model.layer2 = nn.Sequential(
                self.model.layer2,
                nn.Dropout(p=self.dropout_rate)
            )

        if 'layer3' in self.dropout_positions:
            self.model.layer3 = nn.Sequential(
                self.model.layer3,
                nn.Dropout(p=self.dropout_rate)
            )

        if 'layer4' in self.dropout_positions:
            self.model.layer4 = nn.Sequential(
                self.model.layer4,
                nn.Dropout(p=self.dropout_rate)
            )

        if 'avgpool' in self.dropout_positions:
            self.model.avgpool = nn.Sequential(
                self.model.avgpool,
                nn.Dropout(p=self.dropout_rate)
            )

        if 'fc' in self.dropout_positions:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                self.model.fc
            )

    def forward(self, x):
        return self.model(x)