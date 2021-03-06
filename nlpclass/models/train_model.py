"""
Main entry point, reads parameters from the command line and trains the model
"""

import argparse

from nlpclass.models.training_utils import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NLP model')

    parser.add_argument('language', type=str)
    parser.add_argument('--network_type', type=str, default='recurrent')
    parser.add_argument('--attention', action='store_true')

    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers_enc', type=int, default=1)
    parser.add_argument('--num_layers_dec', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--kernel_size', type=int, default=7)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--clipping_value', type=float, default=7.5)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--early_stopping', type=int, default=3)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    parser.add_argument('--pretrained_embeddings', action='store_true')

    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--beam_alpha', type=float, default=1.0)

    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--run_id', type=str, default=None)

    args = parser.parse_args()
    args_dict = vars(args)

    train_model(**args_dict)
