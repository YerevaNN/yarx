import torch
from allennlp.training.metrics import Metric

from overrides import overrides
from typing import Any, Dict, List, Tuple, Set


@Metric.register("relex_mention_recall")
class RelexMentionRecall(Metric):
    def __init__(self) -> None:
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0

    @overrides
    def __call__(self,  # type: ignore
                 batched_top_spans: torch.Tensor,
                 batched_gold_mentions: List[Dict[str, Any]]):
        for top_spans, gold_mentions in zip(batched_top_spans.data.tolist(), batched_gold_mentions):

            gold_mentions: Set[Tuple[int, int]] = set(gold_mentions)
            predicted_spans: Set[Tuple[int, int]] = {(span[0], span[1]) for span in top_spans}
            self._num_gold_mentions += len(gold_mentions)
            self._num_recalled_mentions += len(gold_mentions & predicted_spans)

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        if self._num_gold_mentions == 0:
            recall = 0.0
        else:
            recall = self._num_recalled_mentions/float(self._num_gold_mentions)
        if reset:
            self.reset()
        return recall

    @overrides
    def reset(self):
        """
        Reset any accumulators or internal state.
        """
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
