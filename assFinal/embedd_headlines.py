import utils
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-emb", "--embeddings", default='glove_reviews.json', type=str,
                        help="Embedding file we are using (default glove_reviews.json).\n" +
                        "Has no effect when using pretrained language models.")
    parser.add_argument("-lm", "--language-model", default="bert", type=str,
                        help="Name of pretrained language model to use.\n" +
                        "If not specified will use a LSTM model.")
    parser.add_argument("-ep","--epochs", default=None, type=int,
                        help="Override the default number of epochs to train the model.")
    parser.add_argument("-bs","--batch-size", default=None, type=int,
                        help="Override the default batch size.")
    parser.add_argument("-lr","--learning-rate", default=None, type=float,
                        help="Override the default learning rate.")
    parser.add_argument("--max-length", default=None, type=int,
                        help="Override the default maximum length of language model input.\n" +
                        "Only affects when using language models.")
    parser.add_argument("--strategy", default="cls", type=str,
                        help="The strategy to embedd the sentence\n" +
                        "Can be one of: \"cls\",\"avg\",\"lstm\" or \"bilstm\".")
    parser.add_argument("--freeze", action="store_true",
                        help="If this flag is present the weights of the language model\n" +
                        "are frozen (not updated).")

    args = parser.parse_args()
    return args