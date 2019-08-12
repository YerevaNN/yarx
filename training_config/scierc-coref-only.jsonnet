// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
local dataset_url = 'http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz';
local train_data_path = dataset_url + '#processed_data/json/train.json';
local validation_data_path = dataset_url + '#processed_data/json/dev.json';
local test_data_path = dataset_url + '#processed_data/json/test.json';

// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).

local max_span_width = 8;

local glove_embedding_dim = 300;
local char_embedding_dim = 100;
local elmo_embedding_dim = 0; // 1024; #0;
local bert_embedding_dim = 768;

local embedding_dim = glove_embedding_dim + char_embedding_dim + elmo_embedding_dim + bert_embedding_dim ;
local encoding_dim = 200;
local feedforward_hidden_dim = 150;

local bi_encoding_dim = 2 * encoding_dim;

local feature_size = 20;

local mention_dim = embedding_dim + 2 * bi_encoding_dim + feature_size;
local antecedent_dim = 3 * mention_dim + feature_size;
local relex_span_pair_dim = 3 * mention_dim;



{
  "dataset_reader": {
    "type": "scierc",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 5
      },
//      "elmo": {
//        "type": "elmo_characters"
//      },
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": 'bert-base-multilingual-cased',
          "do_lowercase": false,
          "use_starting_offsets": true,
          "max_pieces": 4096
      }
    },
    "max_span_width": max_span_width,
//    "max_instances_to_read": 10
  },

  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "test_data_path": test_data_path,

  "model": {
    "type": "sciie",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "token_characters": ["token_characters"],
        "tokens": ["tokens"],
//        "elmo": ["elmo"]
      },
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
//        "elmo":{
//            "type": "elmo_token_embedder",
//            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
//            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
//            "do_layer_norm": false,
////            "dropout": 0.5
//        },
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": 'bert-base-multilingual-cased'
        }
      }
    },
    "context_layer": {
        "type": "stacked_bidirectional_lstm",
        "input_size": embedding_dim,
        "hidden_size": encoding_dim,
        "num_layers": 1,
        "layer_dropout_probability": 0.4,
        "recurrent_dropout_probability": 0.4
    },
    "mention_feedforward": {
        "input_dim": mention_dim,
        "num_layers": 2,
        "hidden_dims": feedforward_hidden_dim,
        "activations": "relu",
        "dropout": 0.5
    },
    "relex_mention_feedforward": {
        "input_dim": mention_dim,
        "num_layers": 2,
        "hidden_dims": feedforward_hidden_dim,
        "activations": "relu",
        "dropout": 0.5
    },
    "antecedent_feedforward": {
        "input_dim": antecedent_dim,
        "num_layers": 2,
        "hidden_dims": feedforward_hidden_dim,
        "activations": "relu",
        "dropout": 0.5
    },
    "relex_feedforward": {
        "input_dim": relex_span_pair_dim,
        "num_layers": 2,
        "hidden_dims": feedforward_hidden_dim,
        "activations": "relu",
        "dropout": 0.5
    },
    "initializer": [
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer._module.weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],

        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}],

        ["_context_layer._module.*.input_linearity.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.*.state_linearity.weight", {"type": "orthogonal"}],
    ],
    "lexical_dropout": 0.5,
    "feature_size": feature_size,
    "max_span_width": max_span_width,
    "spans_per_word": 0.4,
    "relex_spans_per_word": 0.4,
    "max_antecedents": 100,
    "loss_coref_weight": 0,
    "loss_relex_weight": 1
  },
  "iterator": {
    "type": "basic",
//    "sorting_keys": [["text", "num_tokens"]],
//    "padding_noise": 0.0,
    "batch_size": 3,
//    "biggest_batch_first": true
  },
  "trainer": {
    "num_epochs": 3000,
//    "grad_norm": 5.0,
    "patience" : 3000,
    "cuda_device" : 0,
    "validation_metric": "+relex_fscore",
//    "learning_rate_scheduler": {
//      "type": "reduce_on_plateau",
//      "factor": 0.5,
//      "mode": "max",
//      "patience": 2
//    },
    "optimizer": {
      "type": "adam"
    },
    "num_serialized_models_to_keep": 4
  }
}