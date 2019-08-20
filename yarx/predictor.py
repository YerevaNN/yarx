import json
from collections import defaultdict
from typing import List, cast, Tuple, Dict, Set

from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import Instance, DatasetReader, Token
from allennlp.data.fields import MetadataField
from allennlp.models import Model
from allennlp.predictors import CorefPredictor, Predictor
from overrides import overrides

from .models.decoder import Cluster


@Predictor.register('sciie')  # for backward compatibility
@Predictor.register('biorelex')
class SciIEPredictor(CorefPredictor):
    """
    Predictor for SciIE model.
    Outputs in BioRelEx format. # TODO support different formats
    """
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 language: str = 'en_core_web_sm') -> None:
        self._decode_clusters = True
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


    def _format_cluster(self, cluster: Cluster):
        names = {}
        for name, mention in cluster.entities:
            if name not in names:
                names[name] = {
                    'is_mentioned': True,
                    'mentions': []
                }
            names[name]['mentions'].append(mention)

        return {
            'is_state': False,
            'label': "TODO",  # TODO
            'is_mentioned': True,
            'is_mutant': False,
            'names': names
        }

    def _format_clusters(self, clusters: List[Cluster]):
        return [self._format_cluster(cluster)
                for cluster in clusters]

    def _format_interaction(self, interaction: Tuple[Tuple[int, int], str]):
        participant_ids, label = interaction

        return {
            'participants': participant_ids,
            'type': label,
            'implicit': False,
            'label': 1
        }

    def _filter_interaction(self, interaction: Tuple[Tuple[int, int], str]):
        participant_ids, label = interaction

        if participant_ids[0] > participant_ids[1]:
            return False

        return True

    def _format_interactions(self, interactions: List[Tuple[Tuple[int, int], str]]):
        return [self._format_interaction(interaction)
                for interaction in interactions
                if self._filter_interaction(interaction)]



    def decode(self, instance: Instance, output_dict):
        metadata = cast(MetadataField, instance['metadata']).metadata

        clusters = output_dict['entities']
        output_dict['entities'] = self._format_clusters(clusters)

        interactions = output_dict['interactions']
        output_dict['interactions'] = self._format_interactions(interactions)


        # We keep id-s in order to make it easy for evaluation stage.
        output_dict['id'] = metadata['id']

        filtered_output = dict()
        for key in output_dict:
            if key not in self._key_whitelist:
                continue
            filtered_output[key] = output_dict[key]

        return filtered_output
