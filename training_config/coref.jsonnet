// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).

local glove_embedding_dim = 300;
local char_embedding_dim = 100;
local elmo_embedding_dim = 1024;

local embedding_dim = glove_embedding_dim + char_embedding_dim + elmo_embedding_dim;
local encoding_dim = 200;
local feedforward_hidden_dim = 150;

local bi_encoding_dim = 2 * encoding_dim;

local feature_size = 20;

local mention_dim = embedding_dim + 2 * bi_encoding_dim + feature_size;
local antecedent_dim = 3 * mention_dim + feature_size;

{
  "dataset_reader": {
    "type": "filter",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 5
      },
      "elmo": {
        "type": "elmo_characters"
      },
//      "bert": {
//          "type": "bert-pretrained",
//          "pretrained_model": 'bert-base-multilingual-cased',
//          "do_lowercase": false,
//          "use_starting_offsets": true,
//          "max_pieces": 4096
//      }
    },
    "max_span_width": 10,
    "max_length": 1024
  },

  "train_data_path": "datasets/train.english.v4_gold_conll",
  "validation_data_path": "datasets/dev.english.v4_gold_conll",
  "test_data_path": "datasets/dev.english.v4_gold_conll",

  "model": {
    "type": "coref",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": glove_embedding_dim,
            "trainable": false
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": char_embedding_dim,
                "ngram_filter_sizes": [5]
            }
        },
        "elmo":{
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.0
        },
//        "bert": {
//            "type": "bert-pretrained",
//            "pretrained_model": 'bert-base-multilingual-cased'
//        }
      }
    },
    "context_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": embedding_dim,
        "hidden_size": encoding_dim,
        "num_layers": 1
    },
    "mention_feedforward": {
        "input_dim": mention_dim,
        "num_layers": 2,
        "hidden_dims": feedforward_hidden_dim,
        "activations": "relu",
        "dropout": 0.2
    },
    "antecedent_feedforward": {
        "input_dim": antecedent_dim,
        "num_layers": 2,
        "hidden_dims": feedforward_hidden_dim,
        "activations": "relu",
        "dropout": 0.2
    },
    "initializer": [
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
    ],
    "lexical_dropout": 0.4,
    "feature_size": feature_size,
    "max_span_width": 10,
    "spans_per_word": 0.4,
    "max_antecedents": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 1,
    "biggest_batch_first": true
  },
  "trainer": {
    "num_epochs": 150,
    "grad_norm": 5.0,
    "patience" : 10,
    "cuda_device" : 0,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam"
    }
  }
}