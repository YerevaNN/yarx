import json
from collections import defaultdict
from typing import List, cast

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.models import Model
from allennlp.predictors import CorefPredictor, Predictor
from overrides import overrides


@Predictor.register('sciie')
class SciIEPredictor(CorefPredictor):
    """
    Predictor for SciIE model.
    Outputs in BioRelEx format. # TODO support different formats
    """
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 language: str = 'en_core_web_sm') -> None:
        self._key_whitelist = [
            'id',
            'text',
            'entities',
            'interactions',
            'clusters'
        ]
        super().__init__(model, dataset_reader, language)

    @overrides
    def predict(self, document: str) -> JsonDict:
        return super().predict(document)

    @overrides
    def predict_tokenized(self, tokenized_document: List[str]) -> JsonDict:
        return super().predict_tokenized(tokenized_document)

    @overrides
    def _words_list_to_instance(self, words: List[str]) -> Instance:
        return super()._words_list_to_instance(words)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return super()._json_to_instance(json_dict)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs) + "\n"

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        outputs = self.decode(instance, outputs)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)

        outputs = [self.decode(instance, output)
                   for instance, output
                   in zip(instances, outputs)]
        return sanitize(outputs)

    def decode(self, instance: Instance, output_dict):
        metadata = cast(MetadataField, instance['metadata']).metadata

        clusters = output_dict['clusters']
        relations = output_dict['relex_predictions']

        span_registry = dict()

        entities = []
        interactions = []

        for cluster in clusters:
            names = defaultdict(lambda: {
                "is_mentioned": True,
                "mentions": []
            })

            for first_idx, last_idx in cluster:
                first = metadata['flat_tokens'][first_idx]
                last = metadata['flat_tokens'][last_idx]

                start = first.idx
                end = last.idx + len(last.text)

                name = metadata['flat_text'][start:end]
                span_registry[start, end] = len(entities)

                names[name]['mentions'].append((start, end))

            entities.append({
                "is_state": False,
                "label": "TODO",  # TODO
                "is_mentioned": True,
                "is_mutant": False,
                "names": names
            })

        for sentence_relations in relations:
            for label, *participants in sentence_relations:
                participant_ids = []
                for first_idx, last_idx in participants:
                    first = metadata['flat_tokens'][first_idx]
                    last = metadata['flat_tokens'][last_idx]

                    start = first.idx
                    end = last.idx + len(last.text)

                    name = metadata['flat_text'][start:end]

                    if (start, end) in span_registry:
                        participant_id = span_registry[start, end]
                    else:
                        participant_id = len(entities)
                        entities.append({
                            "is_state": False,
                            "label": "TODO",
                            "is_mentioned": True,
                            "is_mutant": False,
                            "names": {
                                name: {
                                    "is_mentioned": True,
                                    "mentions": [
                                        [start, end]
                                    ]
                                }
                            }
                        })
                    participant_ids.append(participant_id)

                # The binding relation is symmetric: remove redundant relations
                if participant_ids[0] > participant_ids[1]:
                    continue  # TODO union

                interactions.append({
                    "participants": participant_ids,
                    "type": label,
                    "implicit": False,
                    "label": 1
                })

        output_dict['entities'] = entities
        output_dict['interactions'] = interactions

        # We keep id-s in order to make it easy for evaluation stage.
        output_dict['id'] = metadata['id']

        filtered_output = dict()
        for key in output_dict:
            if key not in self._key_whitelist:
                continue
            filtered_output[key] = output_dict[key]

        return filtered_output
