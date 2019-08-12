# YARX
Yet Another Relation Extraction framework, based on SciIE architecture.

# Getting Started

To start to train the network:
```sh
allennlp train --include-package yarx training_config/YOUR_CONFIG.jsonnet -s logs/SAVE_DIR
```

To continue training:
```sh
allennlp train --include-package yarx logs/SAVE_DIR/config.json -s logs/SAVE_DIR --recover
```

To evaluate on a dataset:
```sh
allennlp evaluate --include-package yarx logs/SAVE_DIR PATH/TO/DATASET.json
```

To run inference:
```sh
allennlp predict --include-package yarx logs/SAVE_DIR --silent PATH/TO/DATASET.json --output-file PATH/TO/OUTPUT.jsonlines --batch-size BATCH_SIZE --use-dataset-reader --predictor sciie
```
