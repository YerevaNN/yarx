import json
import logging
from typing import Iterable, Union
from allennlp.data import DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.data.tokenizers import WordTokenizer

from .scierc import SciERCReader

from overrides import overrides
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("biorelex")
class BioRelExDatasetReader(SciERCReader):
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_instances_to_read: int = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 labels: List[Union[int, str]] = None) -> None:
        """
        DatasetReader for BioRelEx dataset.
        """
        super().__init__(max_span_width, token_indexers, max_instances_to_read, lazy)
        self.tokenizer = tokenizer or WordTokenizer()
        self._positive_labels = labels or [1]

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        for sample in samples:
            yield self.process_sample(sample)

    def tokenize(self, sentence: str):
        tokens = self.tokenizer.tokenize(sentence)

        words: List[str] = []
        starts: Dict[int, int] = dict()
        ends: Dict[int, int] = dict()
        for idx, token in enumerate(tokens):
            words.append(token.text)
            start = token.idx
            # TODO Don't always rely on tokenizer char-based offsets
            # e.g. AllenNLP Bert tokenizer doesn't provide mappings out-of-the-box
            end = start + len(token.text)
            starts[start] = idx
            ends[end] = idx

        mappings = starts, ends

        return tokens, words, mappings

    def process_sample(self, sample: Dict[str, Any]):
        sample_id = sample['id']
        text = sample['text']

        # Here we have to decently tokenize sentences,
        # make them segmented enough so entities does not lie
        # between different tokens.

        # In biorelex, there is only sentence in the sample.
        tokens, words, token_mappings = self.tokenize(text)
        doc = [words]
        doc_tokens = [tokens]
        raw_doc = [text]
        
        token_starts, token_ends = token_mappings

        entities = sample.get('entities')
        if entities is not None:

            # Here we read clusters and make all the interaction pairs
            # Span offsets are with respect to characters.
            raw_clusters: List[List[Tuple[int, int]]] = []
            raw_ner_labels: Dict[Tuple[int, int], str] = dict()
            for entity in entities:
                raw_positions: List[Tuple[int, int]] = list()
                ner_label = entity['label']
                for name in entity['names'].values():
                    for begin, end in name['mentions']:
                        raw_position = begin, end
                        raw_positions.append(raw_position)
                        raw_ner_labels[raw_position] = ner_label
                raw_clusters.append(raw_positions)

            # Now we start mapping between character offsets and
            # token offsets. As we can have spans not matching with
            # our tokenization, some spans are going to be dropped.
            # This is done separately in order to pass character offsets
            # for proper evaluation.
            clusters: List[List[Tuple[int, int]]] = []
            ner_labels: Dict[Tuple[int, int], str] = dict()
            for raw_positions in raw_clusters:
                positions: List[Tuple[int, int]] = []
                for begin, end in raw_positions:
                    first = token_starts.get(begin)
                    last = token_ends.get(end)
                    if first is None or last is None:
                        continue
                    position = (first, last)
                    positions.append(position)
                    ner_labels[position] = raw_ner_labels[begin, end]
                clusters.append(positions)

            doc_ner_labels = [ner_labels]
        else:
            raw_clusters = None
            raw_ner_labels = None
            clusters = None
            ner_labels = None
            doc_ner_labels = None

        interactions = sample.get('interactions')
        if interactions is not None:

            raw_relations: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = dict()
            relations: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = dict()
            for interaction in interactions:
                label = interaction['type']
                a_id, b_id = interaction['participants']

                if interaction['label'] not in self._positive_labels:
                    continue

                # Character-based offsets are stored for fair evaluation
                for a in raw_clusters[a_id]:
                    for b in raw_clusters[b_id]:
                        raw_relations[a, b] = label
                        raw_relations[b, a] = label

                # Token-based offsets are stored for training purposes.
                for a in clusters[a_id]:
                    for b in clusters[b_id]:
                        relations[a, b] = label
                        relations[b, a] = label


            doc_raw_relations = raw_relations
            doc_relations = [relations]
        else:
            relations = None
            doc_raw_relations = None
            doc_relations = None

        metadata = {
            'id': sample_id,
            'doc_tokens': doc_tokens,
            'raw_doc': raw_doc,
            'flat_text': text,
            'flat_tokens': tokens,
            'doc_raw_relations': doc_raw_relations,
            'raw_ner_labels': raw_ner_labels
        }

        if clusters is not None:
            # Filter out clusters having only one mention.
            clusters = [cluster for cluster in clusters if len(cluster) > 1]
        return self.build_instance(doc, clusters,
                                   doc_relations,
                                   doc_ner_labels,
                                   **metadata)

    def text_to_instance(self, *inputs) -> Instance:
        pass
