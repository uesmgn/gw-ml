import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        )
        self.x2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        )
        self.x3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        )

        self.args = None


    def forward(self, x):
        print(x.shape)
        x = x.view(x.shape[0], -1)
        weights = torch.randn((1024, x.shape[-1]))
        x = F.linear(x, weights)
        x = self.x1(x)
        x = self.x2(x)
        x = self.x3(x)
        self.args = {'x': x}
        return x

if __name__ == '__main__':
    x = torch.randn((4, 486, 486))
    model = TestModule()
    x = model(x)
    print(model.args)
