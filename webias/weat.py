# ############################################################################################### #
#                                                                                                 #
# Implements the WEAT score calculation presented in [1].                                         #
#                                                                                                 #
# [1] https://dx.doi.org/10.1126/science.aal4230                                                  #
#                                                                                                 #
# ############################################################################################### #


import logging
import numpy as np

from scipy.spatial.distance import cdist
from tqdm import tqdm

from webias.constants import LOGGING_CONFIG
from webias.utils import build_word_embedding_cache

logging.basicConfig(**LOGGING_CONFIG)


class WEAT:
    """An implementation of the "Word Embeddings Association Test" metric.

    This metric was originally proposed in [1]. For a detailed explanation of the score calculation,
    see the documentation of the `get_score()` method.

    [1] https://dx.doi.org/10.1126/science.aal4230

    Arguments:
    word_vector_getter -- An object that returns a vector given a word as parameter to the
                          `__getitem__()` function.
    """

    def __init__(self, word_vector_getter):
        self.word_vector_getter = word_vector_getter

    def _association_test(
            self,
            word_vectors: np.ndarray,
            attributes_a: np.ndarray,
            attributes_b: np.ndarray) -> float:
        """Calculate the association of a given word vector to the attribute matrices $A$ and $B$.

        Return the association value that resembles the relative similarity between the word and the
        two attribute matrices.

        In the original WEAT paper [1], the calculation is formulated as:
        $s(w, A, B)
            = mean_{a\in A} cos(\vec{w}, \vec{a}) - mean_{b\in B} cos(\vec{w}, \vec{b})$


        [1] https://doi.org/10.1126/science.aal4230

        Arguments:
        word_vector -- The word vectors for which the association should be calculated.
        attributes_a -- Matrix of word vectors for all attribute words in $A$.
        attributes_b -- Matrix of word vectors for all attribute words in $B$.
        """
        association_values_a = np.mean(cdist(word_vectors, attributes_a, metric="cosine"), axis=1)
        association_values_b = np.mean(cdist(word_vectors, attributes_b, metric="cosine"), axis=1)

        return np.subtract(association_values_a, association_values_b) * -1

    def _differential_association_test(
            self,
            word_vectors_X: np.ndarray,
            word_vectors_Y: np.ndarray,
            attributes_a: np.ndarray,
            attributes_b: np.ndarray) -> float:
        """Calculate the difference between the associations of $X$ and $Y$ with $A$ and $B$.

        Return the differential association value that resembles the difference in relative
        similarity between the two target matrices to the two attribute matrices.

        A positive value denotes a closer association between $X$ and $A$, while a negative value
        denotes a closer association between $Y$ and $A$.

        In the original WEAT paper [1], the calculation is formulated as:
        $s(X, Y, A, B) = \sum_{x\in X} s(x, A, B) - \sum_{y\in Y} s(y, A, B)$, where the function
        $s()$ is the association test between a word and two lists of attributes.


        [1] https://dx.doi.org/10.1126/science.aal4230

        Arguments:
        word_vectors_X -- Matrix of word vectors for all target words in $X$.
        word_vectors_Y -- Matrix of word vectors for all target words in $Y$.
        attributes_a -- Matrix of word vectors for all attribute words in $A$.
        attributes_b -- Matrix of word vectors for all attribute words in $B$.
        """
        associations_sum_x = sum(
            self._association_test(word_vectors_X, attributes_a, attributes_b))
        associations_sum_y = sum(
            self._association_test(word_vectors_Y, attributes_a, attributes_b))

        return associations_sum_x - associations_sum_y

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

    def get_score(
            self,
            target_words_X: list,
            target_words_Y: list,
            attribute_words_a: list,
            attribute_words_b: list) -> tuple:
        """Calculates the effect size of the differential association tests.

        Returns a tuple containing the result of the calculation and a list of OOV terms. The score
        simultaniously represents the WEAT score metric and can have values in the range between
        $-2$ and $+2$.

        A positive value denotes a closer association between $X$ and $A$, while a negative value
        denotes a closer association between $Y$ and $A$.

        In the original WEAT paper [1], the calculation of the effect size if formulated as:
        $\frac{mean_{x\in X} s(x, A, B) - mean_{y\in Y} s(y, A, B)}{std\_dev_{w\in X\cup Y}
        s(w, A, B)}$


        [1] https://dx.doi.org/10.1126/science.aal4230

        Arguments:
        target_words_X -- List of target words in $X$.
        target_words_Y -- List of target words in $Y$.
        attribute_words_a -- List of all attribute words in $A$.
        attribute_words_b -- List of all attribute words in $B$.
        """
        # Retrieve all vectors for words in X, Y, A and B
        Xv, oov_x = self._embed_token_list(target_words_X)
        Yv, oov_y = self._embed_token_list(target_words_Y)
        Av, oov_a = self._embed_token_list(attribute_words_a)
        Bv, oov_b = self._embed_token_list(attribute_words_b)

        if len(Xv) == 0 or len(Yv) == 0 or len(Av) == 0 or len(Bv) == 0:
            raise AttributeError("For at least one of the given lexicons all tokens are OOV.")

        # Calculate effect size numerator
        association_X = self._association_test(Xv, Av, Bv)
        association_Y = self._association_test(Yv, Av, Bv)
        numerator = np.mean(association_X) - np.mean(association_Y)

        # Calculate effect size denominator
        denominator = np.std(np.concatenate((association_X, association_Y), axis=0))

        result = numerator / denominator

        return (result, [*oov_x, *oov_y, *oov_a, *oov_b])


def weat_evaluation(
        lexicon: dict,
        embedding_model) -> dict:
    """Evaluate the WEAT metric with the given lexicon.

    Return a dict containing the results of the different shuffled runs per test type.
    Each index in the type list represents the results for one shuffled lexicon (or a shuffled
    lexicon combination if there are two lexicons) in the form of a list. In that list, each index
    `i` represents a lexicon of size `i * m` where `m` is the step size. Thus, it also represents
    the `start_size` ofthe lexicon. Further, `i` also represents the random state used to shuffle
    the original list.

    Arguments:
    lexicon -- The lexicon to be used for the evaluation. Is expected to have the following keys:
               target_set_1, target_set_2, attribute_set_1, attribute_set_2.
    embedding_model -- The embedding model to use for the evaluation.
    """
    word_embedding_cache = build_word_embedding_cache(lexicon, embedding_model)
    # Instantiate the metric with the word vector cache
    weat = WEAT(word_vector_getter=word_embedding_cache)
    # Dict that holds the results
    weat_results = {}

    # For each bias type (e.g. gender, ethnicity, religion) ...
    for test_type, test_lexicons in lexicon.items():
        test_type_results = {
            "shuffled_attribute_results": [],
            "shuffled_target_results": [],
            "attribute_set_lengths": [],
            "target_set_lengths": []}

        # --------------------------------------------------------------------------------
        # Conduct evaluation with shuffled attribute lists

        # Combine the two attribute sets to always use the same index element from both
        shuffled_attribute_runs = list(zip(
            test_lexicons["attribute_set_1"], test_lexicons["attribute_set_2"]))

        attribute_progress_bar = tqdm(shuffled_attribute_runs, desc=f"WEAT-{test_type}-attributes")
        for shuffled_run in attribute_progress_bar:
            shuffled_attribute_run_results = []

            partial_progress_bar = tqdm(list(zip(shuffled_run[0], shuffled_run[1])), leave=False)
            for partial_attribute_1, partial_attribute_2 in partial_progress_bar:
                result = weat.get_score(
                    test_lexicons["target_set_1"][0][-1],
                    test_lexicons["target_set_2"][0][-1],
                    partial_attribute_1,
                    partial_attribute_2)

                shuffled_attribute_run_results.append(result[0])

            # Append information and results to main results object
            test_type_results["shuffled_attribute_results"].append(shuffled_attribute_run_results)

        # Set lengths should be the same across shuffled runs, so we can simply use the lengths of
        # the first one and add them outside the loop
        # But since we have multiple attribuite lists, we need to sum them up at each partial
        attribute_set_lengths = zip(
            map(lambda x: len(x), shuffled_attribute_runs[0][0]),
            map(lambda x: len(x), shuffled_attribute_runs[0][1]))
        test_type_results["attribute_set_lengths"] = [i + j for i, j in attribute_set_lengths]

        # --------------------------------------------------------------------------------
        # Conduct evaluation with shuffled target lists

        # Combine the two target sets to always use the same index element from both
        shuffled_target_runs = list(zip(
            test_lexicons["target_set_1"], test_lexicons["target_set_2"]))

        target_progress_bar = tqdm(shuffled_target_runs, desc=f"WEAT-{test_type}-targets")
        for shuffled_run in target_progress_bar:
            shuffled_target_run_results = []

            partial_progress_bar = tqdm(list(zip(shuffled_run[0], shuffled_run[1])), leave=False)
            for partial_target_1, partial_target_2 in partial_progress_bar:
                result = weat.get_score(
                    partial_target_1,
                    partial_target_2,
                    test_lexicons["attribute_set_1"][0][-1],
                    test_lexicons["attribute_set_2"][0][-1])

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

        weat_results[test_type] = test_type_results

    return weat_results
