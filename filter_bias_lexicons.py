import argparse
import json
import logging

from os import path

from webias.constants import LOGGING_CONFIG
from webias.word_vectors import ConceptNetNumberbatchEmbeddings, CustomEmbeddings, GloVeEmbeddings


def filter_lexicon_by_model(lexicon: list, embedding_model, lowercase: bool = False) -> list:
    """Try to retrieve the vectors for the given lexicon from the given model.

    Return a tuple that contains a list of tokens that do not have any OOV token for the given model
    and the list of the out-of-vocabulary tokens in the form of (filtered_list, oov_tokens).

    Arguments:
    lexicon -- The lexicon that should be filtered as a list of words.
    embedding_model -- The embedding model that should be used to retrieve word vectors.
    lowercase -- Whether the provided model is only trained on lowercased tokens and the lexicon
                 should thus be evaluated on lowercased tokens as well.
    """
    out_of_vocabulary = []
    lexicon_prepared = [t.lower() for t in lexicon] if lowercase else lexicon

    # For each token of the provided lexicon, try to retrieve a vector from the provided model and
    # add it to the OOV list if it was not found
    for index, token in enumerate(lexicon_prepared):
        try:
            embedding_model[token]
        except KeyError:
            out_of_vocabulary.append(lexicon[index])
            logging.info(f"Did not find vector for '{token}'. Removing from lexicon.")

    # Filter the provided lexicon with the OOV terms
    return (
        list(filter(lambda x: x not in out_of_vocabulary, lexicon)),
        out_of_vocabulary)


def main():
    # Read provided lexicons
    with open(args.lexicon_path, "r") as f:
        lexicons = json.load(f)

    # Load embedding models from disk
    logging.info("Loading word embedding model from disk...")
    if args.embedding_model == "glove":
        model = GloVeEmbeddings()
    elif args.embedding_model == "numberbatch1908":
        model = ConceptNetNumberbatchEmbeddings()
    else:
        model = CustomEmbeddings(args.embedding_model)

    # New lexicons dict
    filtered_lexicons = {
        "target_sets": {},
        "attribute_sets": {}}

    # For each type of social bias...
    for target_type, target_groups in lexicons["target_sets"].items():
        filtered_lexicons["target_sets"][target_type] = {}

        # For each target group in the current social bias type...
        for target_group, lexicon in target_groups.items():
            # If the lexicon was already filtered for a different model, it should already have
            # "filtered" tokens that we should also copy to the next lexicon
            filtered_previous = lexicon["filtered"] if "filtered" in lexicon.keys() else []

            new_lexicon, filtered = filter_lexicon_by_model(
                lexicon["set"], model, lowercase=args.lowercase)
            filtered_lexicons["target_sets"][target_type][target_group] = {
                "set": new_lexicon,
                "filtered": [*filtered_previous, *filtered]}

    # For each attribute...
    for attribute, lexicon in lexicons["attribute_sets"].items():
        # If the lexicon was already filtered for a different model, it should already have
        # "filtered" tokens that we should also copy to the next lexicon
        filtered_previous = lexicon["filtered"] if "filtered" in lexicon.keys() else []

        new_lexicon, filtered = filter_lexicon_by_model(
            lexicon["set"], model, lowercase=args.lowercase)
        filtered_lexicons["attribute_sets"][attribute] = {
            "set": new_lexicon,
            "filtered": [*filtered_previous, *filtered]}

    # ------------------------------------------------------------
    # Write results to disk
    original_filename = path.splitext(path.basename(args.lexicon_path))[0]
    if args.output_filename:
        filename = args.output_filename
    else:
        filename = (
            f"{original_filename}--filtered_for_"
            f"{path.basename(args.embedding_model)}.json")
    output_file = path.join(args.output, filename)

    logging.info(f"Writing filtered lexicon to disk at '{output_file}'.")

    with open(output_file, "w") as f:
        json.dump(filtered_lexicons, f)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to filter the social bias lexicon for OOV tokens of the given embeddng model.")

    parser.add_argument(
        "-e",
        "--embedding_model",
        required=True,
        type=str,
        help="Path to a embedding model. It needs to be in the word2vec format, binary or plain.",
        metavar="EMBEDDING_MODEL")
    parser.add_argument(
        "-w",
        "--lexicon_path",
        required=True,
        type=str,
        help="Path to the social bias lexicon file.",
        metavar="BIAS_LEXICON_PATH")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the directory where the result file should be written to.",
        metavar="OUTPUT_DIR")
    parser.add_argument(
        "-f",
        "--output_filename",
        type=str,
        default=None,
        help="If set, the given string will be used as filename for the output file. Otherwise, "
             "the default filename will be used.",
        metavar="OUTPUT_FILENAME")
    parser.add_argument(
        "-l",
        "--lowercase",
        action="store_true",
        help="Whether to lowercase all lexicons before testing or not. This is sometimes required "
             "when the embedding model was generated on solely lowercased tokens.")

    args = parser.parse_args()
    logging.basicConfig(**LOGGING_CONFIG)

    main()
    logging.info("Done.")
