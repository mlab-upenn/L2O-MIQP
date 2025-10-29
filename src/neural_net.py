import torch
import torch.nn as nn

# STE_Round operator; see https://arxiv.org/pdf/1308.3432
class STE_Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Simple_MLP(nn.Module):
    """
    Just a Simple Multi-Layer Perceptron.
    """
    def __init__(self, insize, outsize, bias=True, linear_map=nn.Linear,
        nonlin=nn.ReLU(), hsizes=[64], linargs=None):
        super().__init__()
        if linargs is None:
            linargs = {}

        self.in_features, self.out_features = insize, outsize
        sizes = [insize] + hsizes + [outsize]

        # Define layers
        self.linear = nn.ModuleList([
            linear_map(sizes[i], sizes[i+1], bias=bias, **linargs)
            for i in range(len(sizes) - 1)
        ])

        self.nonlin = nn.ModuleList(
            [nonlin() for _ in range(len(hsizes))] + [nn.Identity()]
        )

    def forward(self, x):
        """
        Forward pass through the network
        """
        for lin, act in zip(self.linear, self.nonlin):
            x = act(lin(x))
        return x

class MLPWithSTE(nn.Module):
    """
    Multi-Layer Perceptron with STE rounding operator on output.
    """
    def __init__(self, insize, outsize, bias=True, linear_map=nn.Linear,
        nonlin=nn.ReLU(), hsizes=[64], linargs=None):
        super().__init__()
        if linargs is None:
            linargs = {}

        self.in_features, self.out_features = insize, outsize
        sizes = [insize] + hsizes + [outsize]

        # Define layers
        self.linear = nn.ModuleList([
            linear_map(sizes[i], sizes[i+1], bias=bias, **linargs)
            for i in range(len(sizes) - 1)
        ])

        self.nonlin = nn.ModuleList(
            [nonlin() for _ in range(len(hsizes))] + [nn.Identity()]
        )

    def reg_error(self):
        """
        Optional L2 regularization hook if using custom linear layers with `reg_error()`.
        """
        return sum(
            [layer.reg_error() for layer in self.linear if hasattr(layer, "reg_error")]
        )

    def forward(self, x):
        """
        Forward pass through the network with STE rounding at the end.
        """
        for lin, act in zip(self.linear, self.nonlin):
            x = act(lin(x))
        probs = torch.sigmoid(x)
        hard_output = STE_Round.apply(probs)
        return hard_output