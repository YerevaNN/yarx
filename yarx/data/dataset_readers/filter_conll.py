from typing import Dict, List, Optional, Tuple

from allennlp.data import DatasetReader, TokenIndexer, Instance
from allennlp.data.dataset_readers import ConllCorefReader
from allennlp.data.fields import TextField


@DatasetReader.register("filter_conll")
class FilterConllCorefReader(ConllCorefReader):
    def __init__(self,
                 *,
                 max_length: int = 512,
                 validation_mode: bool = False,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False):
        # validation mode replaces long samples with dummy ones
        super().__init__(max_span_width, token_indexers, lazy)
        self._max_length = max_length
        self._validation_mode = validation_mode

    def _read(self, file_path: str):
        for sample in super()._read(file_path):
            text_field: TextField = sample.fields["text"]
            if not self._validation_mode and len(text_field.tokens) > self._max_length:
                continue
            yield sample

    def text_to_instance(self,
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None) -> Instance:
        return super().text_to_instance(sentences, gold_clusters)

