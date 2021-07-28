# ############################################################################################### #
#                                                                                                 #
# Implements the ECT score calculation presented in [1].                                          #
#                                                                                                 #
# [1] http://proceedings.mlr.press/v89/dev19a.html                                                #
#                                                                                                 #
# ############################################################################################### #

import logging
import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from tqdm import tqdm

from webias.constants import LOGGING_CONFIG
from webias.utils import build_word_embedding_cache

logging.basicConfig(**LOGGING_CONFIG)


class ECT:
    """Implementation of the Embedding Coherence Test metric.

    The metrics was originally proposed in [1] and implemented in [2]. The general process of the
    metric, as defined in [1], works as follows:

    1. Embedd all given target and attribute words with the given embedding model
    2. Calculate mean vectors for the two sets of target word vectors
    3. Measure the cosine similarity of the mean target vectors to all of the given attribute words
    4. Calculate the Spearman r correlation between the resulting two lists of similarities
    5. Return the correlation value as score of the metric (in the range of -1 to 1); higher is
    better

    [1]: Dev, et al. 2019: "Attenuating Bias in Word Vectors"
    [2]: https://github.com/sunipa/Attenuating-Bias-in-Word-Vec

    Arguments:
    word_vector_getter -- An object that returns a vector given a word as parameter to the
                          `__getitem__()` function.
    """

    def __init__(self, word_vector_getter):
        self.word_vector_getter = word_vector_getter

    def _embed_token_list(self, token_list: list) -> tuple:
        """Transform a list of tokens to a list of word vectors. Return the list.

        If a token is found to be out-of-vocabulary, it will be added to a separate list that is
        returned alongside the list of vectors; the token will be excluded from the latter.

        Arguments:
        token_list -- A list of tokens that should be transformed.
        """
        vector_list = []
        oov = []
        for token in token_list:
            try:
                vector_list.append(self.word_vector_getter[token])
            except KeyError:
                logging.debug(f"Token '{token}' is OOV. Ignoring.")
                oov.append(token)

        return (vector_list, oov)

    def get_score(self, target_set_1: list, target_set_2: list, attribute_set: list) -> tuple:
        """Calculate the ECT score for the given word sets.

        Return the correlation alongside its p-value and the OOV tokens as a tuple in the form of
        `(result, pvalue, OOV)`.

        Arguments:
        target_set_1 -- A list of words representing the first set of target words.
        target_set_2 -- A list of words representing the second set of target words.
        attribute_set -- A list of words representing the set of attribute words.
        """

        target_vectors_1, oov_1 = self._embed_token_list(target_set_1)
        target_vectors_2, oov_2 = self._embed_token_list(target_set_2)
        attribute_vectors, oov_a = self._embed_token_list(attribute_set)

        # Calculate mean vectors for both target vector sets
        target_means = [np.mean(s, axis=0) for s in (target_vectors_1, target_vectors_2)]

        # Measure similarities between mean vectors and all attribute words
        similarities = 1 - cdist(target_means, attribute_vectors, metric="cosine")

        # Calculate similarity correlations
        result = spearmanr(similarities[0], similarities[1])
        return (result.correlation, result.pvalue, [*oov_1, *oov_2, *oov_a])


def ect_evaluation(
        lexicon: dict,
        embedding_model) -> dict:
    """Evaluate the ECT metric with the given lexicon.

    Return a dict containing the results of the different shuffled runs per test type.
    Each index in the type list represents the results for one shuffled lexicon (or a shuffled
    lexicon combination if there are two lexicons) in the form of a list. In that list, each index
    `i` represents a lexicon of size `i * m` where `m` is the step size. Thus, it also represents
    the `start_size` ofthe lexicon. Further, `i` also represents the random state used to shuffle
    the original list.

    Arguments:
    lexicon -- The lexicon to be used for the evaluation. Is expected to have the following keys:
               target_set_1, target_set_2, attribute_set.
    embedding_model -- The embedding model to use for the evaluation.
    """
    word_embedding_cache = build_word_embedding_cache(lexicon, embedding_model)
    # Instantiate the metric with the word vector cache
    ect = ECT(word_vector_getter=word_embedding_cache)
    # Dict that holds results
    ect_results = {}

    # For each bias type (e.g. gender, ethnicity, religion) ...
    for test_type, test_lexicons in lexicon.items():
        test_type_results = {
            "shuffled_attribute_results": [],
            "shuffled_target_results": [],
            "attribute_set_lengths": [],
            "target_set_lengths": []}

        # --------------------------------------------------------------------------------
        # Conduct evaluation with shuffled attribute list

        attributes_progress_bar = tqdm(
            test_lexicons["attribute_set"],
            desc=f"ECT-{test_type}-attributes")
        for shuffled_run in attributes_progress_bar:
            shuffled_attribute_run_results = []

            # For the target sets we can use the full set of the first shuffled run, since
            # we are using always the same and order is not important here
            for partial_run in tqdm(shuffled_run, leave=False):
                result = ect.get_score(
                    test_lexicons["target_set_1"][0][-1],
                    test_lexicons["target_set_2"][0][-1],
                    partial_run)

                shuffled_attribute_run_results.append(result[0])

            # Append information and results to main results objects
            test_type_results["shuffled_attribute_results"].append(shuffled_attribute_run_results)

        # Set lengths should be the same across shuffled runs, so we can simply use the lengths of
        # the first one and add them outside the loop
        test_type_results["attribute_set_lengths"] = list(
            map(lambda x: len(x), test_lexicons["attribute_set"][0]))

        # --------------------------------------------------------------------------------
        # Conduct evaluation with shuffled target lists

        # Combine the two target sets to always use the same index element from both
        shuffled_target_runs = list(zip(
            test_lexicons["target_set_1"], test_lexicons["target_set_2"]))

        target_progress_bar = tqdm(shuffled_target_runs, desc=f"ECT-{test_type}-targets")
        for shuffled_run in target_progress_bar:
            shuffled_target_run_results = []

            partial_progress_bar = tqdm(list(zip(shuffled_run[0], shuffled_run[1])), leave=False)
            for partial_target_1, partial_target_2 in partial_progress_bar:
                result = ect.get_score(
                    partial_target_1,
                    partial_target_2,
                    test_lexicons["attribute_set"][0][-1])

                shuffled_target_run_results.append(result[0])

            # Append information and results to main results object
            test_type_results["shuffled_target_results"].append(shuffled_target_run_results)

        # Set lengths should be the same across shuffled runs, so we can simply use the lengths of
        # the first one and add them outside the loop
        # But since we have multiple target lists, we need to sum them up at each partial
        target_set_lengths = zip(
            map(lambda x: len(x), shuffled_target_runs[0][0]),
            map(lambda x: len(x), shuffled_target_runs[0][1]))
        test_type_results["target_set_lengths"] = [i + j for i, j in target_set_lengths]

        ect_results[test_type] = test_type_results

    return ect_results
