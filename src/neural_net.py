import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax

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


class MLPWithSoftmaxSTE(nn.Module):
    """
    MLP whose output integers are selected via straight-through Gumbel-Softmax.
    """

    def __init__(
        self,
        insize,
        outsize,
        integer_choices=None,
        bias=True,
        linear_map=nn.Linear,
        nonlin=nn.ReLU,
        hsizes=None,
        tau=2.0,
        hard=True,
        use_gumbel=True,
        linargs=None,
    ):
        super().__init__()
        if linargs is None:
            linargs = {}
        if hsizes is None:
            hsizes = [64]

        self.in_features = insize
        self.out_features = outsize
        self.tau = tau
        self.hard = hard
        self.use_gumbel = use_gumbel

        if integer_choices is None:
            integer_choices = torch.arange(0, 2, dtype=torch.get_default_dtype())

        integer_tensor = torch.as_tensor(integer_choices, dtype=torch.get_default_dtype())
        if integer_tensor.dim() == 1:
            integer_tensor = integer_tensor.unsqueeze(0).repeat(self.out_features, 1)
        elif integer_tensor.dim() == 2:
            if integer_tensor.shape[0] != self.out_features:
                raise ValueError(
                    "When `integer_choices` is 2D, its first dimension must equal `outsize`."
                )
        else:
            raise ValueError("`integer_choices` must be 1D or 2D tensor-like.")

        self.num_integers = integer_tensor.shape[-1]
        sizes = [insize] + hsizes + [self.out_features * self.num_integers]

        self.linear = nn.ModuleList(
            [
                linear_map(sizes[i], sizes[i + 1], bias=bias, **linargs)
                for i in range(len(sizes) - 1)
            ]
        )
        self.nonlin = nn.ModuleList(
            [nonlin() for _ in range(len(hsizes))] + [nn.Identity()]
        )

        self.register_buffer(
            "integer_buffer",
            integer_tensor.reshape(1, self.out_features, self.num_integers),
            persistent=False,
        )

    def forward(self, x):
        """
        Forward pass returning straight-through rounded integers.
        """
        for lin, act in zip(self.linear, self.nonlin):
            x = act(lin(x))

        logits = x.view(*x.shape[:-1], self.out_features, self.num_integers)
        flat_logits = logits.reshape(-1, self.num_integers)
        if self.use_gumbel:
            one_hot = gumbel_softmax(flat_logits, tau=self.tau, hard=self.hard)
        else:
            soft = torch.softmax(flat_logits / self.tau, dim=-1)
            if self.hard:
                indices = soft.argmax(dim=-1)
                hard_one_hot = torch.nn.functional.one_hot(
                    indices, num_classes=self.num_integers
                ).type_as(soft)
                one_hot = hard_one_hot - hard_one_hot.detach() + soft
            else:
                one_hot = soft

        one_hot = one_hot.view(*logits.shape)
        integers = (one_hot * self.integer_buffer).sum(dim=-1)
        return integers.view(*x.shape[:-1], self.out_features)
