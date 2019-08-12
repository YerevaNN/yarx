import torch
from torch.nn import functional as F

from allennlp.modules import TimeDistributed

from typing import List

from overrides import overrides


class MultiTimeDistributed(TimeDistributed):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.

    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.

    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional ``pass_through`` iterable.


    This implementation differs from the regular TimeDistributes with its ability of handling
    modules returning multiple outputs.
    """

    @overrides
    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):
        # pylint: disable=arguments-differ
        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        return self._reshape_outputs_back(reshaped_outputs, some_input)

    @classmethod
    def _reshape_outputs_back(cls,
                              reshaped_outputs,
                              some_input: torch.Tensor):

        if isinstance(reshaped_outputs, (list, tuple)):
            outputs = [
                cls._reshape_outputs_back(reshaped_output, some_input)
                for reshaped_output in reshaped_outputs
            ]
            if isinstance(reshaped_outputs, tuple):
                outputs = tuple(outputs)
            return outputs

        if isinstance(reshaped_outputs, dict):
            return {
                key: cls._reshape_outputs_back(reshaped_output, some_input)
                for key, reshaped_output in reshaped_outputs.items()
            }

        return cls._reshape_tensor_back(reshaped_outputs, some_input)

    @staticmethod
    def _reshape_tensor_back(reshaped_output: torch.Tensor,
                             some_input: torch.Tensor) -> torch.Tensor:
        new_size = some_input.size()[:2] + reshaped_output.size()[1:]
        return reshaped_output.contiguous().view(new_size)
