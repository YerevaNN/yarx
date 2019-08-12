import torch
from torch.nn import functional as F
from overrides import overrides


class Projection(torch.nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int = 1,
                 bias: bool = True,
                 with_dummy: bool = False):
        """
        A simple extension of Linear layer with support of dummy neuron.
        """
        self._with_dummy = with_dummy
        if not with_dummy:
            super().__init__(in_features, out_features, bias)
        else:
            super().__init__(in_features, out_features - 1, bias)

    @overrides
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = super().forward(input)

        if not self._with_dummy:
            return outputs

        padding = (1, 0)  # pad with least index, do not pad the last
        return F.pad(outputs, padding)

    @overrides
    def extra_repr(self):
        extra_repr = super().extra_repr()

        if not self._with_dummy:
            extra_repr = f'with_dummy, {extra_repr}'

        return extra_repr
