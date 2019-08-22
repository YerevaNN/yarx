from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
from allennlp.training.metrics import Metric

from typing import Any, Dict, List, Tuple, Union, Iterable


@Metric.register('precision_recall_fscore')
class PrecisionRecallFScore(Metric):
    def __init__(self,
                 beta: float = 1,
                 labels: Iterable[str] = None,
                 average: str = None,
                 pos_label: str = None,
                 verbose: bool = True) -> None:
        self._beta = beta
        self._average = average
        self._labels = list(labels)
        self._pos_label = pos_label
        self._verbose = verbose

        self._current_predictions: List[List[Any]] = None
        self._current_labels: List[List[Any]] = None

        self.reset()

    def __call__(self,
                 predictions: List[Any],
                 gold_labels: List[Any]):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A set of predictions.
        gold_labels : ``torch.Tensor``, required.
            A set of ground truth.
        """
        self._current_predictions.append(predictions)
        self._current_labels.append(gold_labels)

    def get_metric(self,
                   reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """

        # Flatten
        y_true = [y for ys in self._current_labels for y in ys]
        y_pred = [y for ys in self._current_predictions for y in ys]

        scores = self._get_metric_on(y_true, y_pred)

        if reset:
            self.reset()
            return scores

        # Report incomplete scores only if `verbose` is set
        if self._verbose:
            return scores

        # Wait for `reset=True` to report complete scores.
        return dict()


    def _get_metric_on(self, y_true, y_pred):
        # TODO Against multiple sub-splits
        # TODO Evaluate with our own code, optimize
        prfs = precision_recall_fscore_support(y_true,
                                               y_pred,
                                               beta=self._beta,
                                               labels=self._labels,
                                               average=self._average,
                                               pos_label=self._pos_label)
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=self._labels)

        # Now we have precision, recall and f_beta score for all the labels
        # We are going to report scores for all the labels besides null-label.
        scores = dict()
        for class_name, *class_prfs, class_cm in zip(self._labels, *prfs, cm):
            precision, recall, fscore, support = class_prfs
            (tn, fp), (fn, tp) = class_cm

            if not class_name:
                continue

            scores[f'{class_name}_precision'] = float(precision)
            scores[f'{class_name}_recall'] = float(recall)
            scores[f'{class_name}_f{self._beta}'] = float(fscore)
            scores[f'{class_name}_support'] = int(support)

            scores[f'{class_name}_tn'] = int(tn)
            scores[f'{class_name}_fp'] = int(fp)
            # scores[f'{class_name}_fn'] = int(fn)
            scores[f'{class_name}_tp'] = int(tp)


        return scores

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self._current_predictions = []
        self._current_labels = []
