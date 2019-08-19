# YARX
Yet Another Relation Extraction framework, based on [SciIE architecture](http://nlp.cs.washington.edu/sciIE/).

# Getting Started

To start to train the network:
```sh
python -m yarx train training_config/YOUR_CONFIG.jsonnet -s logs/SAVE_DIR
```

To continue training:
```sh
python -m yarx train logs/SAVE_DIR/config.json -s logs/SAVE_DIR --recover
```

To evaluate on a dataset:
```sh
python -m yarx evaluate logs/SAVE_DIR PATH/TO/DATASET.json
```

To run inference:
```sh
python -m yarx predict logs/SAVE_DIR --silent PATH/TO/DATASET.json --output-file PATH/TO/OUTPUT.jsonlines --batch-size BATCH_SIZE --use-dataset-reader --predictor sciie
```
