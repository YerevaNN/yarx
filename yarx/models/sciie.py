import math
import logging
from collections import defaultdict

import torch
from allennlp.models import CoreferenceResolver
from torch.nn import Sequential
from torch.nn import functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, Pruner
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator


from overrides import overrides
from typing import Any, Dict, List, Optional, Tuple, Set


from ..layers import MultiTimeDistributed, Projection
from ..metrics import PrecisionRecallFScore, RelexMentionRecall
from ..utils import nd_batched_index_select, nd_batched_padded_index_select, nd_cross_entropy_with_logits

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("sciie")
class SciIE(CoreferenceResolver):
    """
    This ``Model`` implements the coreference resolution model described "End-to-end Neural
    Coreference Resolution"
    <https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
    by Lee et al., 2017.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 relex_feedforward: FeedForward,
                 antecedent_feedforward: FeedForward,
                 feature_size: int,
                 max_span_width: int,
                 spans_per_word: float,
                 relex_spans_per_word: float,
                 max_antecedents: int,
                 mention_feedforward: FeedForward,
                 coref_mention_feedforward: FeedForward = None,
                 relex_mention_feedforward: FeedForward = None,
                 symmetric_relations: bool = False,
                 lexical_dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 loss_coref_weight: float = 1,
                 loss_relex_weight: float = 1,
                 loss_ner_weight: float = 1,
                 preserve_metadata: List = None,
                 relex_namespace: str = 'relation_labels') -> None:
        # If separate coref mention and relex mention feedforward scorers
        # are not provided, share the one of NER module
        if coref_mention_feedforward is None:
            coref_mention_feedforward = mention_feedforward
        if relex_mention_feedforward is None:
            relex_mention_feedforward = mention_feedforward

        super().__init__(vocab, text_field_embedder,
                         context_layer, coref_mention_feedforward,
                         antecedent_feedforward, feature_size,
                         max_span_width, spans_per_word, max_antecedents,
                         lexical_dropout, initializer, regularizer)

        self._symmetric_relations = symmetric_relations
        self._relex_spans_per_word = relex_spans_per_word
        self._loss_coref_weight = loss_coref_weight
        self._loss_relex_weight = loss_relex_weight
        self._loss_ner_weight = loss_ner_weight
        self._preserve_metadata = preserve_metadata or ['id']
        self._relex_namespace = relex_namespace

        relex_labels = list(vocab.get_token_to_index_vocabulary(self._relex_namespace))
        self._relex_mention_recall = RelexMentionRecall()
        self._relex_precision_recall_fscore = PrecisionRecallFScore(labels=relex_labels)

        relex_mention_scorer = Sequential(
            TimeDistributed(relex_mention_feedforward),
            TimeDistributed(Projection(relex_mention_feedforward.get_output_dim()))
        )
        self._relex_mention_pruner = MultiTimeDistributed(Pruner(relex_mention_scorer))

        self._ner_scorer = Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(Projection(mention_feedforward.get_output_dim(),
                                       vocab.get_vocab_size('ner_labels'),
                                       with_dummy=True))
        )

        self._relex_scorer = Sequential(
            TimeDistributed(relex_feedforward),
            TimeDistributed(Projection(relex_feedforward.get_output_dim(),
                                       vocab.get_vocab_size(self._relex_namespace),
                                       with_dummy=True))
        )

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                metadata: List[Dict[str, Any]],
                doc_span_offsets: torch.IntTensor,
                span_labels: torch.IntTensor = None,
                doc_truth_spans: torch.IntTensor = None,
                doc_spans_in_truth: torch.IntTensor = None,
                doc_relation_labels: torch.Tensor = None,
                truth_spans: List[Set[Tuple[int, int]]] = None,
                doc_relations = None,
                doc_ner_labels: torch.IntTensor = None,
                ) -> Dict[str, torch.Tensor]:  # add matrix from datareader
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : ``Dict[str, torch.LongTensor]``, required.
            The output of a ``TextField`` representing the text of
            the document.
        spans : ``torch.IntTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a ``ListField[SpanField]`` of
            indices into the text of the document.
        span_labels : ``torch.IntTensor``, optional (default = None)
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        metadata : ``torch.IntTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a ``ListField[SpanField]`` of
            indices into the text of the document.
        doc_ner_labels : ``torch.IntTensor``.
            A tensor of shape # TODO,
            ...
        doc_span_offsets : ``torch.IntTensor``.
            A tensor of shape (batch_size, max_sentences, max_spans_per_sentence, 1),
            ...
        doc_truth_spans : ``torch.IntTensor``.
            A tensor of shape (batch_size, max_sentences, max_truth_spans, 1),
            ...
        doc_spans_in_truth : ``torch.IntTensor``.
            A tensor of shape (batch_size, max_sentences, max_spans_per_sentence, 1),
            ...
        doc_relation_labels : ``torch.Tensor``.
            A tensor of shape (batch_size, max_sentences, max_truth_spans, max_truth_spans),
            ...

        Returns
        -------
        An output dictionary consisting of:
        top_spans : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep, 2)`` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : ``torch.IntTensor``
            A tensor of shape ``(num_spans_to_keep, max_antecedents)`` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep)`` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        batch_size = len(spans)
        document_length = text_embeddings.size(1)
        max_sentence_length = max(len(sentence)
                                  for document in metadata
                                  for sentence in document['doc_tokens'])
        num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # TODO features dropout
        # Shape: (batch_size, num_spans, embedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))
        num_relex_spans_to_keep = int(math.floor(self._relex_spans_per_word * max_sentence_length))

        # Shapes:
        # (batch_size, num_spans_to_keep, span_dim),
        # (batch_size, num_spans_to_keep),
        # (batch_size, num_spans_to_keep),
        # (batch_size, num_spans_to_keep, 1)
        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores) = self._mention_pruner(span_embeddings,
                                                                           span_mask,
                                                                           num_spans_to_keep)
        # Shape: (batch_size, num_spans_to_keep, 1)
        top_span_mask = top_span_mask.unsqueeze(-1)

        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Now that we have our variables in terms of num_spans_to_keep, we need to
        # compare span pairs to decide each span's antecedent. Each span can only
        # have prior spans as antecedents, and we only consider up to max_antecedents
        # prior spans. So the first thing we do is construct a matrix mapping a span's
        #  index to the indices of its allowed antecedents. Note that this is independent
        #  of the batch dimension - it's just a function of the span's position in
        # top_spans. The spans are in document order, so we can just use the relative
        # index of the spans to know which other spans are allowed antecedents.

        # Once we have this matrix, we reformat our variables again to get embeddings
        # for all valid antecedents for each span. This gives us variables with shapes
        #  like (batch_size, num_spans_to_keep, max_antecedents, embedding_size), which
        #  we can use to make coreference decisions between valid span pairs.

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
            self._generate_valid_antecedents(num_spans_to_keep, max_antecedents, util.get_device_of(text_mask))
        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings,
                                                                      valid_antecedent_indices)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores,
                                                                          valid_antecedent_indices).squeeze(-1)
        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings,
                                                                  candidate_antecedent_embeddings,
                                                                  valid_antecedent_offsets)
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(span_pair_embeddings,
                                                              top_span_mention_scores,
                                                              candidate_antecedent_mention_scores,
                                                              valid_antecedent_log_mask)

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1


        output_dict = dict()

        output_dict["top_spans"] = top_spans
        output_dict["antecedent_indices"] = valid_antecedent_indices
        output_dict["predicted_antecedents"] = predicted_antecedents


        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]

        # Shape: (,)
        loss = 0

        # Shape: (batch_size, max_sentences, max_spans)
        doc_span_mask = (doc_span_offsets[:, :, :, 0] >= 0).float()
        # Shape: (batch_size, max_sentences, num_spans, span_dim)
        doc_span_embeddings = util.batched_index_select(span_embeddings,
                                                        doc_span_offsets.squeeze(-1).long().clamp(min=0))


        # Shapes:
        # (batch_size, max_sentences, num_relex_spans_to_keep, span_dim),
        # (batch_size, max_sentences, num_relex_spans_to_keep),
        # (batch_size, max_sentences, num_relex_spans_to_keep),
        # (batch_size, max_sentences, num_relex_spans_to_keep, 1)
        pruned = self._relex_mention_pruner(doc_span_embeddings,
                                            doc_span_mask,
                                            num_items_to_keep=num_relex_spans_to_keep,
                                            pass_through=['num_items_to_keep'])
        (top_relex_span_embeddings, top_relex_span_mask,
         top_relex_span_indices, top_relex_span_mention_scores) = pruned

        # Shape: (batch_size, max_sentences, num_relex_spans_to_keep, 1)
        top_relex_span_mask = top_relex_span_mask.unsqueeze(-1)

        # Shape: (batch_size, max_sentences, max_spans_per_sentence, 2)  # TODO do we need for a mask?
        doc_spans = util.batched_index_select(spans, doc_span_offsets.clamp(0).squeeze(-1))

        # Shape: (batch_size, max_sentences, num_relex_spans_to_keep, 2)
        top_relex_spans = nd_batched_index_select(doc_spans, top_relex_span_indices)

        # Shapes:
        # (batch_size, max_sentences, num_relex_spans_to_keep, num_relex_spans_to_keep, 3 * span_dim),
        # (batch_size, max_sentences, num_relex_spans_to_keep, num_relex_spans_to_keep).
        (relex_span_pair_embeddings,
         relex_span_pair_mask) = self._compute_relex_span_pair_embeddings(top_relex_span_embeddings,
                                                                          top_relex_span_mask.squeeze(-1))

        # Shape: (batch_size, max_sentences, num_relex_spans_to_keep, num_relex_spans_to_keep, num_relation_labels)
        relex_scores = self._compute_relex_scores(relex_span_pair_embeddings,
                                                  top_relex_span_mention_scores)
        output_dict['relex_scores'] = relex_scores
        output_dict['top_relex_spans'] = top_relex_spans


        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(span_labels.unsqueeze(-1),
                                                           top_span_indices,
                                                           flat_top_span_indices)
            antecedent_labels_ = util.flattened_index_select(pruned_gold_labels,
                                                             valid_antecedent_indices).squeeze(-1)
            antecedent_labels = antecedent_labels_ + valid_antecedent_log_mask.long()

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels,
                                                                          antecedent_labels)
            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability x to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.
            coreference_log_probs = util.masked_log_softmax(coreference_scores, top_span_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs)
            negative_marginal_log_likelihood *= top_span_mask.squeeze(-1).float()
            negative_marginal_log_likelihood = negative_marginal_log_likelihood.sum()

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(top_spans, valid_antecedent_indices, predicted_antecedents, metadata)

            coref_loss = negative_marginal_log_likelihood
            output_dict['coref_loss'] = coref_loss
            loss += self._loss_coref_weight * coref_loss


        if doc_relations is not None:

            # The adjacency matrix for relation extraction is very sparse.
            # As it is not just sparse, but row/column sparse (only few
            # rows and columns are non-zero and in that case these rows/columns
            # are not sparse), we implemented our own matrix for the case.
            # Here we have indices of truth spans and mapping, using which
            # we map prediction matrix on truth matrix.
            # TODO Add teacher forcing support.

            # Shape: (batch_size, max_sentences, num_relex_spans_to_keep),
            relative_indices = top_relex_span_indices
            # Shape: (batch_size, max_sentences, num_relex_spans_to_keep, 1),
            compressed_indices = nd_batched_padded_index_select(doc_spans_in_truth,
                                                                relative_indices)

            # Shape: (batch_size, max_sentences, num_relex_spans_to_keep, max_truth_spans)
            gold_pruned_rows = nd_batched_padded_index_select(doc_relation_labels,
                                                              compressed_indices.squeeze(-1),
                                                              padding_value=0)
            gold_pruned_rows = gold_pruned_rows.permute(0, 1, 3, 2).contiguous()

            # Shape: (batch_size, max_sentences, num_relex_spans_to_keep, num_relex_spans_to_keep)
            gold_pruned_matrices = nd_batched_padded_index_select(gold_pruned_rows,
                                                                  compressed_indices.squeeze(-1),
                                                                  padding_value=0)  # pad with epsilon
            gold_pruned_matrices = gold_pruned_matrices.permute(0, 1, 3, 2).contiguous()


            # TODO log_mask relex score before passing
            relex_loss = nd_cross_entropy_with_logits(relex_scores,
                                                      gold_pruned_matrices,
                                                      relex_span_pair_mask)
            output_dict['relex_loss'] = relex_loss

            self._relex_mention_recall(top_relex_spans.view(batch_size, -1, 2),
                                       truth_spans)
            self._compute_relex_metrics(output_dict, doc_relations)

            loss += self._loss_relex_weight * relex_loss


        if doc_ner_labels is not None:
            # Shape: (batch_size, max_sentences, num_spans, num_ner_classes)
            ner_scores = self._ner_scorer(doc_span_embeddings)
            output_dict['ner_scores'] = ner_scores

            ner_loss = nd_cross_entropy_with_logits(ner_scores,
                                                    doc_ner_labels,
                                                    doc_span_mask)
            output_dict['ner_loss'] = ner_loss
            loss += self._loss_ner_weight * ner_loss


        if not isinstance(loss, int):  # If loss is not yet modified
            output_dict["loss"] = loss

        return output_dict

    def _compute_relex_metrics(self, output_dict, doc_relations):
        """
        Compute Relation Extraction metrics. Note, this is not the same as
        in original BioRelEx task evaluation which is a bit complicated.
        """
        output_dict = self.decode(output_dict)  # To calculate F1 score, we need to to call decode step
        batch_predicted_relations = output_dict['relex_predictions']

        for sample_predictions, sample_relations in zip(batch_predicted_relations,
                                                        doc_relations):
            for sentence_predictions, sentence_relations in zip(sample_predictions,
                                                                sample_relations):
                candidates = defaultdict(lambda: defaultdict(str))

                for predicted_label, a, b in sentence_predictions:
                    candidates[a, b]['prediction'] = predicted_label

                for true_label, a, b in sentence_relations:
                    candidates[a, b]['label'] = true_label

                predicted_labels = [candidate['prediction'] for candidate in candidates.values()]
                true_labels = [candidate['label'] for candidate in candidates.values()]

                self._relex_precision_recall_fscore(predicted_labels, true_labels)

    @overrides
    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode coreference and relation extraction parts.
        """

        # If the predictions are already decoded, skip the step
        if 'relex_predictions' in output_dict:
            return output_dict

        output_dict = super().decode(output_dict)

        relex_predictions = self._decode_relex_predictions(output_dict['relex_scores'],
                                                           output_dict['top_relex_spans'])
        output_dict['relex_predictions'] = relex_predictions
        return output_dict

    def _decode_relex_predictions(self, batch_scores, batch_spans):
        # Transfer tensors to cpu to make interaction faster
        batch_scores = batch_scores.detach().cpu()
        batch_spans = batch_spans.detach().cpu()

        batch_predictions = []
        for doc_scores, doc_spans in zip(batch_scores, batch_spans):
            doc_predictions = []
            for scores, spans in zip(doc_scores, doc_spans):
                sentence_predictions = []

                predicted_labels = scores.argmax(-1)
                nonzero_indices = predicted_labels.nonzero()

                if len(nonzero_indices) == 0:
                    continue

                numerical_labels = predicted_labels[tuple(nonzero_indices.t())].tolist()  # TODO simplify
                relations = spans[nonzero_indices].tolist()
                for numerical_label, relation in zip(numerical_labels, relations):
                    first_span = tuple(relation[0])
                    second_span = tuple(relation[1])
                    label = self.vocab.get_token_from_index(numerical_label,
                                                            self._relex_namespace)
                    relation = (label, first_span, second_span)
                    sentence_predictions.append(relation)

                doc_predictions.append(sentence_predictions)
            batch_predictions.append(doc_predictions)

        return batch_predictions


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns a dictionary of metrics.
        """
        metrics = super().get_metrics(reset)

        metrics['relex_mention_recall'] = self._relex_mention_recall.get_metric(reset)

        relex_precision_recall_fscore = self._relex_precision_recall_fscore.get_metric(reset)
        metrics.update(relex_precision_recall_fscore)

        return metrics

    def _compute_relex_span_pair_embeddings(self,
                                            top_relex_span_embeddings: torch.FloatTensor,
                                            top_relex_span_mask: torch.Tensor
                                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes an embedding representation of pairs of spans for the pairwise
        scoring function to consider. This includes both the original span
        representations and the element-wise similarity of the span representations.

        Parameters
        ----------
        top_relex_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, *, num_spans_to_keep, span_dim).
        top_relex_span_mask : ``torch.Tensor``, required.
            Mask of the given spans. Has shape
            (batch_size, *, num_spans_to_keep).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, *, num_spans_to_keep, num_spans_to_keep, span_dim)
        span_pair_mask : ``torch.FloatTensor``
            Log mask for span pair representations. Has shape
            (batch_size, *, num_spans_to_keep, num_spans_to_keep)
        """

        # Shape: (batch_size, *, num_spans_to_keep, 1, span_dim)
        first_span = top_relex_span_embeddings.unsqueeze(-2)
        # Shape: (batch_size, *, 1, num_spans_to_keep, span_dim)
        second_span = top_relex_span_embeddings.unsqueeze(-3)

        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep, span_dim)
        similarity = first_span * second_span

        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep, span_dim)
        first_span = first_span.expand_as(similarity)
        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep, span_dim)
        second_span = second_span.expand_as(similarity)

        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep, 3 * span_dim)
        span_pair_embeddings = torch.cat([first_span,
                                          second_span,
                                          similarity], -1)

        # Shape: (batch_size, *, num_spans_to_keep, 1)
        first_span_mask = top_relex_span_mask.unsqueeze(-1)
        # Shape: (batch_size, *, 1, num_spans_to_keep)
        second_span_mask = top_relex_span_mask.unsqueeze(-2)
        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep)
        span_pair_mask = first_span_mask * second_span_mask

        return span_pair_embeddings, span_pair_mask

    @staticmethod
    def _compute_antecedent_gold_labels(top_span_labels: torch.IntTensor,
                                        antecedent_labels: torch.IntTensor):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        Parameters
        ----------
        top_span_labels : ``torch.IntTensor``, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : ``torch.IntTensor``, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        Returns
        -------
        pairwise_labels_with_dummy_label : ``torch.FloatTensor``
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_relex_scores(self,
                              pairwise_embeddings: torch.FloatTensor,
                              mention_scores: torch.FloatTensor,
                              # log_mask: torch.FloatTensor
                              ) -> torch.FloatTensor:
        """
        Compute score matrix of Relation Extraction task.
        """

        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep, num_relation_labels)
        relex_scores = self._relex_scorer(pairwise_embeddings)

        if self._symmetric_relations:
            relex_scores += relex_scores.transpose(-2, -3)

        # Shape: (batch_size, *, num_spans_to_keep, 1, 1).
        first_mention_scores = mention_scores.unsqueeze(-2)
        # Shape: (batch_size, *, 1, num_spans_to_keep, 1).
        second_mention_scores = mention_scores.unsqueeze(-3)

        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep, num_relation_labels)
        relex_scores += first_mention_scores + second_mention_scores
        # Shape: (batch_size, *, num_spans_to_keep, num_spans_to_keep, num_relation_labels)
        # relex_scores += log_mask

        # Zero dummy-scores
        relex_scores[..., 0] = 0

        return relex_scores
