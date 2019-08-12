import torch
from allennlp.nn import util
from typing import Any


def nd_batched_padded_index_select(target: torch.Tensor,
                                   indices: torch.IntTensor,
                                   padding_value: Any = -1):
    """
    Multidimensional version of `util.batched_index_select`,
    supports also padded indices: for padded indices the corresponding
    return values are padded with `padding_value`.
    """
    mask = (indices >= 0).unsqueeze(-1).float()
    indices = indices.clamp(min=0)

    res = nd_batched_index_select(target, indices)

    return util.replace_masked_values(res, mask, padding_value)


def nd_batched_index_select(target: torch.Tensor,
                            indices: torch.IntTensor) -> torch.Tensor:
    """
    Multidimensional version of `util.batched_index_select`.
    """
    batch_axes = target.size()[:-2]
    num_batch_axes = len(batch_axes)
    target_shape = target.size()
    indices_shape = indices.size()

    target_reshaped = target.view(-1, *target_shape[num_batch_axes:])
    indices_reshaped = indices.view(-1, *indices_shape[num_batch_axes:])

    output_reshaped = util.batched_index_select(target_reshaped, indices_reshaped)

    return output_reshaped.view(*indices_shape, -1)


def nd_cross_entropy_with_logits(logits: torch.Tensor,
                                 targets: torch.Tensor,
                                 weights: torch.Tensor):
    """
    Multidimensional version of `util.cross_entropy_with_logits`.

    # Shape: (batch_size, d_1, ..., d_n, num_classes)
    logits
    # Shape: (batch_size, d_1, ..., d_n)
    targets
    # Shape: (batch_size, d_1, ..., d_n)
    weights
    """

    # Shape : (batch_size * d_1 * ... * d_n, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # Shape : (batch_size * d_1 * ... * d_n, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # Shape : (batch_size * d_1 * ... * d_n, 1)
    targets_flat = targets.view(-1, 1).long().clamp(0)

    # Contribution to the negative log likelihood only comes from the exact indices
    # of the targets, as the target distributions are one-hot. Here we use torch.gather
    # to extract the indices of the num_classes dimension which contribute to the loss.

    # Shape : (batch_size * d_1 * ... * d_n, 1)
    negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # Shape : (batch_size, d_1, ..., d_n)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # Shape : (batch_size, d_1, ..., d_n)
    loss = negative_log_likelihood * weights.float()
    # Shape : (batch_size, d_1, ..., d_n)
    mask = weights
    # Shape : (batch_size, d_1, ..., d_n-1)
    while loss.dim() > 0:
        mask = (mask > 0).sum(-1)
        # shape : (batch_size, d_1, ..., d_n-1)
        loss = loss.sum(-1) / (mask.float() + 1e-13)

    return loss
