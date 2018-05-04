import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, input_size, class_count):
        super(LogisticRegression, self).__init__()
        self.main = nn.Linear(input_size, class_count)

    def forward(self, x):
        return self.main(x)
