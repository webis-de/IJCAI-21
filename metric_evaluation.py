import argparse
import json
import logging
import math

from datetime import datetime
from os import path

from webias.constants import LOGGING_CONFIG
from webias.ect import ect_evaluation
from webias.rnsb import rnsb_evaluation
from webias.weat import weat_evaluation
from webias.word_vectors import ConceptNetNumberbatchEmbeddings, CustomEmbeddings, GloVeEmbeddings
from webias.utils import prepare_lexicons, prepare_combined_lexicons


def main():
    # Check attribute and target step size
    # Print warning if required
    if args.attribute_step_size % 2 > 0 or args.target_step_size % 2 > 0:
        logging.warning(
            "Note that you provided a step size that makes it impossible to increase two lists by "
            "an equal amount at each step. This is due to the fact that the provided step size is "
            "not divisible by two. The script will floor the provided step sizes in cases where "
            "two lists need to be enlarged.")

    # Load lexicons from disk
    logging.info("Reading lexicon from disk.")
    with open(args.lexicon_path, "r") as f:
        lexicons = json.load(f)

    # Load embedding models from disk
    logging.info("Loading embedding model.")
    if args.embedding_model == "glove":
        model = GloVeEmbeddings()
    elif args.embedding_model == "numberbatch1908":
        model = ConceptNetNumberbatchEmbeddings()
    else:
        model = CustomEmbeddings(args.embedding_model)

    # ------------------------------------------------------------
    # Prepare all lexicons
    gender_male_set, gender_female_set = prepare_lexicons(
        lexicon_1=lexicons["target_sets"]["gender"]["male"]["set"],
        lexicon_2=lexicons["target_sets"]["gender"]["female"]["set"],
        shuffled_runs=args.shuffled_runs,
        step_size=math.floor(args.target_step_size / 2),
        lowercase=args.lowercase,
        allow_different_lengths=args.allow_different_lexicon_sizes)
    ethnicity_euam_set, ethnicity_afam_set = prepare_lexicons(
        lexicon_1=lexicons["target_sets"]["ethnicity"]["european_american"]["set"],
        lexicon_2=lexicons["target_sets"]["ethnicity"]["african_american"]["set"],
        shuffled_runs=args.shuffled_runs,
        step_size=math.floor(args.target_step_size / 2),
        lowercase=args.lowercase,
        allow_different_lengths=args.allow_different_lexicon_sizes)
    religion_chris_set, religion_isla_set = prepare_lexicons(
        lexicon_1=lexicons["target_sets"]["religion"]["christianity"]["set"],
        lexicon_2=lexicons["target_sets"]["religion"]["islam"]["set"],
        shuffled_runs=args.shuffled_runs,
        step_size=math.floor(args.target_step_size / 2),
        lowercase=args.lowercase,
        allow_different_lengths=args.allow_different_lexicon_sizes)

    # Attribute sets
    profession_male_set, profession_female_set = prepare_lexicons(
        lexicon_1=lexicons["attribute_sets"]["male_professions"]["set"],
        lexicon_2=lexicons["attribute_sets"]["female_professions"]["set"],
        shuffled_runs=args.shuffled_runs,
        step_size=math.floor(args.attribute_step_size / 2),
        lowercase=args.lowercase,
        allow_different_lengths=args.allow_different_lexicon_sizes)
    sentiment_positive_set, sentiment_negative_set = prepare_lexicons(
        lexicon_1=lexicons["attribute_sets"]["positive"]["set"],
        lexicon_2=lexicons["attribute_sets"]["negative"]["set"],
        shuffled_runs=args.shuffled_runs,
        step_size=math.floor(args.attribute_step_size / 2),
        lowercase=args.lowercase,
        allow_different_lengths=args.allow_different_lexicon_sizes)

    # Mixed attribute set for metric that only use a single attribute set
    professions_male_female_set = prepare_combined_lexicons(
        lexicon_1=lexicons["attribute_sets"]["male_professions"]["set"],
        lexicon_2=lexicons["attribute_sets"]["female_professions"]["set"],
        shuffled_runs=args.shuffled_runs,
        step_size=math.floor(args.attribute_step_size / 2),
        lowercase=args.lowercase,
        allow_different_lengths=args.allow_different_lexicon_sizes)
    sentiments_positive_negative_set = prepare_combined_lexicons(
        lexicon_1=lexicons["attribute_sets"]["positive"]["set"],
        lexicon_2=lexicons["attribute_sets"]["negative"]["set"],
        shuffled_runs=args.shuffled_runs,
        step_size=math.floor(args.attribute_step_size / 2),
        lowercase=args.lowercase,
        allow_different_lengths=args.allow_different_lexicon_sizes)

    # ------------------------------------------------------------
    # ECT metric evaluation
    ect_lexicons = {
        "gender": {
            "target_set_1": gender_male_set,
            "target_set_2": gender_female_set,
            "attribute_set": professions_male_female_set},
        "ethnicity": {
            "target_set_1": ethnicity_euam_set,
            "target_set_2": ethnicity_afam_set,
            "attribute_set": sentiments_positive_negative_set},
        "religion": {
            "target_set_1": religion_chris_set,
            "target_set_2": religion_isla_set,
            "attribute_set": sentiments_positive_negative_set}}

    logging.info("Starting ECT evaluation.")

    # Initialize ECT test and conduct evaluation
    ect_results = ect_evaluation(
        ect_lexicons,
        model)

    # ------------------------------------------------------------
    # WEAT metric evaluation
    weat_lexicons = {
        "gender": {
            "target_set_1": gender_male_set,
            "target_set_2": gender_female_set,
            "attribute_set_1": profession_male_set,
            "attribute_set_2": profession_female_set},
        "ethnicity": {
            "target_set_1": ethnicity_euam_set,
            "target_set_2": ethnicity_afam_set,
            "attribute_set_1": sentiment_positive_set,
            "attribute_set_2": sentiment_negative_set},
        "religion": {
            "target_set_1": religion_chris_set,
            "target_set_2": religion_isla_set,
            "attribute_set_1": sentiment_positive_set,
            "attribute_set_2": sentiment_negative_set}}

    logging.info("Starting WEAT evaluation.")

    # Initialize WEAT tests and conduct evaluation
    weat_results = weat_evaluation(
        weat_lexicons,
        model)

    # ------------------------------------------------------------
    # RNSB metric evaluation
    rnsb_lexicons = {
        "gender": {
            "target_set_1": gender_male_set,
            "target_set_2": gender_female_set,
            "attribute_set_1": profession_male_set,
            "attribute_set_2": profession_female_set},
        "ethnicity": {
            "target_set_1": ethnicity_euam_set,
            "target_set_2": ethnicity_afam_set,
            "attribute_set_1": sentiment_positive_set,
            "attribute_set_2": sentiment_negative_set},
        "religion": {
            "target_set_1": religion_chris_set,
            "target_set_2": religion_isla_set,
            "attribute_set_1": sentiment_positive_set,
            "attribute_set_2": sentiment_negative_set}}

    logging.info("Starting RNSB evaluation.")

    # Conduct RNSB evaluations
    rnsb_results = {}
    rnsb_results = rnsb_evaluation(
        model,
        rnsb_lexicons)

    # ------------------------------------------------------------
    # Write results to disc
    dt = datetime.today().strftime("%Y%m%d%H%M%S")
    filename = args.output_filename if args.output_filename else f"metric-evaluation-{dt}.json"
    output_file = path.join(args.output, filename)

    logging.info(f"Writing results to disk at '{output_file}'.")

    with open(output_file, "w") as f:
        results = {
            "model_name": path.basename(args.embedding_model),
            "ect": ect_results,
            "weat": weat_results,
            "rnsb": rnsb_results}
        json.dump(results, f)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to evaluate the word embedding bias metrics.")

    parser.add_argument(
        "-e",
        "--embedding_model",
        required=True,
        type=str,
        help="If 'glove', this script will use the 840B CommonCrawl GloVe embeddings. Otherwhise: "
             "path to a embedding model that needs to be in the word2vec format, binary or plain.",
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
    parser.add_argument(
        "-s",
        "--shuffled_runs",
        required=True,
        type=int,
        help="The number of shuffled word lists to evaluate.",
        metavar="SHUFFLED_RUNS")
    parser.add_argument(
        "-a",
        "--attribute_step_size",
        required=True,
        type=int,
        help="The step size with which to increase the size of the attribute word list(s). Also "
             "resembles the size of the list(s) to start with. For metrics with two attribute "
             "lists this number will be divided by 2.",
        metavar="ATTRIBUTE_STEP_SIZE")
    parser.add_argument(
        "-t",
        "--target_step_size",
        required=True,
        type=int,
        help="The step size with which to increase the size of the target word list(s). Also "
             "resembles the size of the list(s) to start with. For metrics with two target lists, "
             "this number will be divided by 2.",
        metavar="TARGET_STEP_SIZE")
    parser.add_argument(
        "-d",
        "--allow_different_lexicon_sizes",
        action="store_true",
        help="Whether to allow for differently sized lexicons to be used. If not passed, all "
             "lexicons will be trimmed to the size of the smallest one. This affects only the "
             "currently shuffled lists.")

    args = parser.parse_args()
    logging.basicConfig(**LOGGING_CONFIG)

    main()

    logging.info("Done.")
