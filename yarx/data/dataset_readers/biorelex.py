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
        self.labels = labels or [1]

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
        sentences = [words]
        sentences_tokens = [tokens]
        raw_sentences = [text]
        
        token_starts, token_ends = token_mappings

        entities = sample.get('entities')
        if entities is not None:
            clusters: List[List[Tuple[int, int]]] = []
            ner_labels: List[Tuple[Tuple[int, int], str]] = []
            for idx, entity in enumerate(entities):
                cluster = set()
                ner_label = entity['label']
                for name, mentions in entity['names'].items():
                    for begin, end in mentions['mentions']:
                        first = token_starts.get(begin)
                        last = token_ends.get(end)
                        if first is None or last is None:
                            continue
                        position = (first, last)
                        cluster.add(position)
                        ner_labels.append((position, ner_label))

                clusters.append(list(cluster))

            sentences_ner_labels = [ner_labels]
        else:
            clusters = None
            ner_labels = None
            sentences_ner_labels = None

        interactions = sample.get('interactions')
        if interactions is not None:
            relations: List[Tuple[str, Tuple[int, int], Tuple[int, int]]] = []
            for interaction in sample['interactions']:
                # in biorelex, the relations are symmetrical
                label = interaction['type']
                a_id, b_id = interaction['participants']
                if interaction['label'] not in self.labels:
                    continue
                for a in clusters[a_id]:
                    for b in clusters[b_id]:
                        relations.append((label, a, b))
                        relations.append((label, b, a))
            sentences_relations = [relations]
        else:
            relations = None
            sentences_relations = None

        metadata = {
            'id': sample_id,
            'sentences_tokens': sentences_tokens,
            'raw_sentences': raw_sentences,
            'flat_text': text,
            'flat_tokens': tokens
        }

        if clusters is not None:
            # Filter out clusters having only one mention.
            clusters = [cluster for cluster in clusters if len(cluster) > 1]
        return self.build_instance(sentences, clusters,
                                   sentences_relations,
                                   sentences_ner_labels,
                                   **metadata)

    def text_to_instance(self, *inputs) -> Instance:
        pass
