import collections
import itertools
import json
import logging
import tarfile

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Field
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, SpanField, ListField, MetadataField, SequenceLabelField, IndexField, \
    AdjacencyField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from ..fields import OptionalListField

from overrides import overrides
from typing import Dict, Any, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("scierc")
class SciERCReader(DatasetReader):
    """
    DatasetReader for SciERC dataset.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_instances_to_read: int = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_instances_to_read = max_instances_to_read
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    # todo multiprocessdatasetreader compatible

    def _iter_file_lines(self, file_path: str, encoding: str = 'utf-8'):
        file_path, _, inner_path = file_path.partition('#')
        file_path = cached_path(file_path)

        if not inner_path:
            with open(file_path, 'r', encoding=encoding) as f:
                yield from f
            return

        with tarfile.open(file_path, 'r') as tar:
            for line in tar.extractfile(inner_path):
                yield line.decode('utf-8')

    @overrides
    def _read(self, file_path: str):
        for idx, line in enumerate(self._iter_file_lines(file_path)):
            if self._max_instances_to_read is not None and idx >= self._max_instances_to_read:
                break

            line = line.strip()
            if line:
                yield self._read_line(line)

    def _read_line(self, line: str) -> Instance:
        sample = json.loads(line)

        sentences: List[List[str]] = sample['sentences']
        clusters: List[List[Tuple[int, int]]] = [
            [(start, end) for start, end in cluster]
            for cluster in sample['clusters']
        ]

        relations: List[Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]] = [
            {
                (tuple(relation[0:2]), tuple(relation[2:4])): relation[-1]
                for relation in sentence
            } for sentence in sample['relations']
        ]

        instance = self.build_instance(sentences, clusters, relations)

        return instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         doc: List[List[str]],
                         clusters: Optional[List[List[Tuple[int, int]]]] = None,
                         doc_relations: List[Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]] = None) -> Instance:
        return self.build_instance(doc, clusters, doc_relations)

    # noinspection PyTypeChecker
    def build_instance(self,  # type: ignore
                       doc: List[List[str]],
                       clusters: List[List[Tuple[int, int]]] = None,
                       doc_relations: List[Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]] = None,
                       doc_ner_labels: List[Dict[Tuple[int, int], str]] = None,
                       **kwargs) -> Instance:
        """
        Parameters
        ----------
        doc : ``List[List[str]]``, required.
            A list of lists representing the tokenized words and sentences in the document.
        clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.
        doc_relations : TODO

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.

        Extra fields:

            spans : see docstring
                Shape:  (num_spans)
                 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

            sentences_span_indices : list spans (absolute indices) for every sentence,
                can be used in order to isolate spans between different sentences.
                By design, the RelEx part considers intra-sentences truth_relations
                and is able to extract inter-sentence truth_relations with the help
                of already predicted sets of coreferences.

                Shape: (sentences_padded, spans_in_sentence_padded)
                 0   1   2   3
                 4   5   6   #
                 7   8   9   #
                10  11  12  13
                14  15   #   #
                Range: [0, ..., num_spans-1], # is padding

            TODO
            sentences_truth_spans : relative indices in sentence_spans
                correspond to at least one relation from truth.
                Intended to be used for effective packing and padding of
                the sparse matrix.

                PyTorch lacks of (at least stable) support of sparse tensors,
                and we aim to implement it ourselves. The matrix is not going
                to be encoded using COO because the sparsity of matrix is just
                an effect of sparsity of the truth spans. This matrix is simply
                compressed matrix w.r.t. COO-encoded spans that are going
                to be used for encoding the relation matrix.

                Shape: (sentences_padded, gold_spans_in_sentence_padded)
                 1   3
                 0   2
                 0   1
                 2   #
                 1   #
                Range: [0, ..., spans_in_sentence - 1], # is padding

            TODO
            sentences_spans_in_truth : simply the inverse of `sentences_truth_spans`
                This matrix can be also interpreted as boolean matrix of if the
                span occurs is truth span: if the element is not padded, it is,
                and the element points out where they occur in compressed matrix.

                Shape: (sentences_padded, spans_in_sentence_padded)
                 #   0   #   1
                 0   #   1   #
                 0   1   #   #
                 #   #   0   #
                 #   0   #   #
                Range: [0, ..., gold_spans_in_sentence_padded - 1], # is padding

            TODO
            sentences_relations : TODO

                Shape: (sentences_padded, gold_spans_in_sentence_padded, gold_spans_in_sentence_padded)
                Range: [0, ..., num_classes - 1], # is padding

            sentences_ner_labels : TODO

                Shape: TODO
                Range: TODO

        """

        metadatas: Dict[str, Any] = {}

        flattened_doc = [self._normalize_word(word)
                         for sentence in doc
                         for word in sentence]
        metadatas["doc_tokens"] = doc
        metadatas["original_text"] = flattened_doc

        metadatas.update(kwargs)

        text_field = TextField([
            Token(word) for word in flattened_doc
        ], self._token_indexers)


        spans: List[SpanField] = []
        doc_span_offsets: List[List[int]] = []

        # Construct spans and mappings
        sentence_offset = 0
        for sentence in doc:
            sentence_spans: List[int] = []

            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                absolute_index = len(spans)
                spans.append(SpanField(start, end, text_field))
                sentence_spans.append(absolute_index)

            sentence_offset += len(sentence)
            doc_span_offsets.append(sentence_spans)

        # Just making fields out of the lists
        spans_field = OptionalListField(spans, empty_field=SpanField(-1, -1, text_field).empty_field())
        doc_span_offsets_field = ListField([
            OptionalListField([
                IndexField(span_offset, spans_field)
                for span_offset in sentence_span_offsets
            ], empty_field=IndexField(-1, spans_field).empty_field())
            for sentence_span_offsets in doc_span_offsets
        ])



        # num_sentences = len(sentences)
        # num_spans = len(spans)
        # inverse_mapping = -np.ones(shape=(num_sentences, num_spans), dtype=int)
        # for sentence_id, indices in enumerate(sentences_span_indices):
        #     for gold_index, real_index in enumerate(indices.array):
        #         inverse_mapping[sentence_id, real_index] = gold_index



        # sentences_spans_field = ListField([
        #     ListField(spans) for spans in sentences_span_indices
        # ])
        # sentences_span_inverse_mapping_field = ArrayField(inverse_mapping, padding_value=-1)

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": spans_field,
            "doc_span_offsets": doc_span_offsets_field
        }

        # TODO TODO TODO rename sentences to doc, sencence to snt

        for key, value in metadatas.items():
            fields[key] = MetadataField(value)

        if clusters is None or doc_relations is None:
            return Instance(fields)

        # Here we can be sure both `clusters` and `doc_relations` are given.
        # However, we can be sure yet whether `doc_ner_labels` is given or not.

        #
        #               TRUTH AFTER THIS ONLY
        #

        fields["clusters"] = MetadataField(clusters)
        cluster_dict = {
            (start, end): cluster_id
            for cluster_id, cluster in enumerate(clusters)
            for start, end in cluster
        }

        truth_spans = {span
                       for sentence in doc_relations
                       for spans, label in sentence.items()
                       for span in spans}
        fields["truth_spans"] = MetadataField(truth_spans)


        span_labels: Optional[List[int]] = []
        doc_truth_spans: List[List[int]] = []
        doc_spans_in_truth: List[List[int]] = []

        for sentence, sentence_spans_field in zip(doc, doc_span_offsets_field):
            sentence_truth_spans: List[IndexField] = []
            sentence_spans_in_truth: List[int] = []

            for relative_index, span in enumerate(sentence_spans_field):
                absolute_index = cast(IndexField, span).sequence_index
                span_field: SpanField = cast(SpanField, spans_field[absolute_index])

                start = span_field.span_start
                end = span_field.span_end

                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

                compressed_index = -1
                if (start, end) in truth_spans:
                    compressed_index = len(sentence_truth_spans)
                    sentence_truth_spans.append(IndexField(relative_index, sentence_spans_field))

                sentence_spans_in_truth.append(compressed_index)

            sentence_truth_spans_field = OptionalListField(sentence_truth_spans,
                                                           empty_field=IndexField(-1, sentence_spans_field).empty_field())
            doc_truth_spans.append(sentence_truth_spans_field)

            sentence_spans_in_truth_field = OptionalListField([
                IndexField(compressed_index, sentence_truth_spans_field)
                for compressed_index in sentence_spans_in_truth
            ], empty_field=IndexField(-1, sentence_truth_spans_field).empty_field())
            doc_spans_in_truth.append(sentence_spans_in_truth_field)

        span_labels_field = SequenceLabelField(span_labels, spans_field)
        doc_truth_spans_field = ListField(doc_truth_spans)
        doc_spans_in_truth_field = ListField(doc_spans_in_truth)

        fields["span_labels"] = span_labels_field
        fields["doc_truth_spans"] = doc_truth_spans_field
        fields["doc_spans_in_truth"] = doc_spans_in_truth_field


        # "sentences_span_inverse_mapping": sentences_span_inverse_mapping_field,
        # "truth_relations": MetadataField(truth_relations)

        # our code

        # test code
        # sample_label = LabelField('foo')
        # sample_list = ListField([sample_label,  sample_label])
        # sample_seq_labels = SequenceLabelField(labels=['bar', 'baz'],
        #                                        sequence_field=sample_list)
        #
        # empty_seq_labels = sample_seq_labels.empty_field()


        # TODO reverse matrix generation tactic
        # TODO Add dummy
        doc_relex_matrices: List[AdjacencyField] = []
        for (sentence,
             truth_relations,
             sentence_spans,
             truth_spans_field,
             spans_in_truth) in zip(doc,
                                    doc_relations,
                                    doc_span_offsets,
                                    doc_truth_spans_field,
                                    doc_spans_in_truth_field):

            relations = collections.defaultdict(str)
            for (span_a, span_b), label in truth_relations.items():
                # Span absolute indices (document-wide indexing)
                try:
                    a_absolute_index = spans.index(span_a)
                    b_absolute_index = spans.index(span_b)
                    # Fill the dict as sparse matrix, padded with zeros
                    relations[a_absolute_index, b_absolute_index] = label
                except ValueError:
                    logger.warning('Span not found')

            indices: List[Tuple[int, int]] = []
            labels: List[str] = []

            for span_a, span_b in itertools.product(enumerate(truth_spans_field), repeat=2):
                a_compressed_index, a_relative = cast(Tuple[int, IndexField], span_a)
                b_compressed_index, b_relative = cast(Tuple[int, IndexField], span_b)

                a_absolute = sentence_spans[a_relative.sequence_index]
                b_absolute = sentence_spans[b_relative.sequence_index]

                label = relations[a_absolute, b_absolute]

                indices.append((a_compressed_index, b_compressed_index))
                labels.append(label)


            doc_relex_matrices.append(AdjacencyField(
                indices=indices,
                labels=labels,
                sequence_field=truth_spans_field,
                label_namespace="relation_labels"
            ))  # TODO pad with zeros maybe?

        # fields["doc_relations"] = MetadataField(doc_relations)
        fields["doc_relation_labels"] = ListField(doc_relex_matrices)



        # gold_candidates = []
        # gold_candidate_labels = []
        #
        # for sentence in sentences_relations:
        #
        #     candidates: List[ListField[SpanField]] = []
        #     candidate_labels: List[LabelField] = []
        #
        #     for label, (a_start, a_end), (b_start, b_end) in sentence:
        #         a_span = SpanField(a_start, a_end, text_field)
        #         b_span = SpanField(b_start, b_end, text_field)
        #         candidate_field = ListField([a_span, b_span])
        #         label_field = OptionalLabelField(label, 'relation_labels')
        #
        #         candidates.append(candidate_field)
        #         candidate_labels.append(label_field)
        #
        #     # if not candidates:
        #     #     continue
        #     #     # TODO very very tmp
        #
        #     empty_text = text_field.empty_field()
        #     empty_span = SpanField(-1, -1, empty_text).empty_field()
        #     empty_candidate = ListField([empty_span, empty_span]).empty_field()
        #     empty_candidates = ListField([empty_candidate]).empty_field()
        #     empty_label = OptionalLabelField('', 'relation_labels')  # .empty_field()?
        #     empty_candidate_labels = ListField([empty_label])  # ? .empty_field() ?
        #
        #     if candidates:
        #         candidates_field = ListField(candidates)
        #         candidate_labels_field = ListField(candidate_labels)
        #     else:
        #         candidates_field = empty_candidates
        #         candidate_labels_field = empty_candidate_labels
        #
        #     gold_candidates.append(candidates_field)
        #     gold_candidate_labels.append(candidate_labels_field)
        #
        # fields["gold_candidates"] = ListField(gold_candidates)
        # fields["gold_candidate_labels"] = ListField(gold_candidate_labels)
        #
        # fields["sentences_relations"] = MetadataField(sentences_relations)


        if doc_ner_labels is None:
            return Instance(fields)


        # NER
        doc_ner: List[OptionalListField[LabelField]] = []

        sentence_offset = 0
        for sentence, sentence_ner_dict in zip(doc, doc_ner_labels):
            sentence_ner_labels: List[LabelField] = []

            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if (start, end) in sentence_ner_dict:
                    label = sentence_ner_dict[(start, end)]
                    sentence_ner_labels.append(LabelField(label, 'ner_labels'))
                else:
                    sentence_ner_labels.append(LabelField('O', 'ner_labels'))

            sentence_offset += len(sentence)
            sentence_ner_labels_field = OptionalListField(sentence_ner_labels,
                                                          empty_field=LabelField('*', 'ner_tags').empty_field())
            doc_ner.append(sentence_ner_labels_field)

        doc_ner_field = ListField(doc_ner)
        fields["doc_ner_labels"] = doc_ner_field

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        # TODO check WTF?
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
