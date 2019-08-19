import argparse
import json

import logging
import os
import sys
from typing import Dict, Any

from allennlp import commands
from allennlp.commands import Subcommand
from allennlp.common import Params
from allennlp.common.util import prepare_environment
from allennlp.data import DatasetReader, DataIterator
from allennlp.models import load_archive
from allennlp.training.util import evaluate

# if os.environ.get("ALLENNLP_DEBUG"):
#     LEVEL = logging.DEBUG
# else:
#     LEVEL = logging.INFO
#
# sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
#                     level=LEVEL)


import yarx

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EvaluateDetailed(Subcommand):
    def add_subparser(self,
                      name: str,
                      parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + for each sample in dataset'''
        subparser = parser.add_parser(
                name, description=description, help='Evaluate the specified model + dataset.')

        subparser.add_argument('archive_file', type=str, help='path to an archived trained model')

        subparser.add_argument('input_file', type=str, help='path to the file containing the evaluation data')

        subparser.add_argument('--output-file', type=str, help='path to output file')

        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--batch-weight-key',
                               type=str,
                               default="",
                               help='If non-empty, name of metric used to weight the loss on a per-batch basis.')

        subparser.add_argument('--extend-vocab',
                               action='store_true',
                               default=False,
                               help='if specified, we will use the instances in your new dataset to '
                                    'extend your vocabulary. If pretrained-file was used to initialize '
                                    'embedding layers, you may also need to pass --embedding-sources-mapping.')

        subparser.add_argument('--embedding-sources-mapping',
                               type=str,
                               default="",
                               help='a JSON dict defining mapping from embedding module path to embedding'
                               'pretrained-file used during training. If not passed, and embedding needs to be '
                               'extended, we will try to use the original file paths used during training. If '
                               'they are not available we will use random vectors for embedding extension.')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    embedding_sources: Dict[str, str] = (json.loads(args.embedding_sources_mapping)
                                         if args.embedding_sources_mapping else {})
    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(Params({}), instances=instances)
        model.extend_embedder_vocab(embedding_sources)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

    keys = None
    for instance in instances:
        metrics = evaluate(model, [instance], iterator, args.cuda_device, args.batch_weight_key)

        if keys is None:
            keys = sorted(metrics.keys())
            print('instance_id', end='')
            for key in keys:
                print(',', end='')
                print(key, end='')
            print('')

        instance_id = instance.fields['metadata']['id']
        print(instance_id, end='')
        for key in keys:
            print(',', end='')
            print(metrics[key], end='')
        print('')

    # output_file = args.output_file
    # if output_file:
    #     with open(output_file, "w") as file:
    #         json.dump(metrics, file, indent=4)
    # return metrics


if __name__ == '__main__':
    commands.main('python -m yarx', {
        'evaluate-detailed': EvaluateDetailed()
    })
