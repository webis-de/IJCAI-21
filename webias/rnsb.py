# ############################################################################################### #
#                                                                                                 #
# Implements the RNSB score calculation as presented in [1].                                      #
#                                                                                                 #
# [1] dx.doi.org/10.18653/v1/P19-1162                                                             #
#                                                                                                 #
# ############################################################################################### #

import logging
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from tqdm import tqdm

from webias.constants import LOGGING_CONFIG
from webias.utils import build_word_embedding_cache

logging.basicConfig(**LOGGING_CONFIG)


class RNSB:
    """An implementation of the "Relative Negative Sentiment Bias" metric.

    This metric was originally proposed in [1]. For a detailed explanation of the score calculation,
    see the documentation of the `get_score()` method.

    [1] dx.doi.org/10.18653/v1/P19-1162

    Arguments:
    positive_words -- A list of positive words used to train the classifier.
    negative_words -- A list of negative words used to train the classifier.
    positive_vectors -- A list of word vectors from positive words used to train the classifier.
                        They are expected to come from the same embedding model given as
                        `word_vector_getter`. This can be used as an alternative to the
                        `positive_words` to speed up the calculation if the class is re-generated
                        multiple times. Also, this can only be used together with `negative_vectors`
                        and not with `negative_words`.
    negative_vectors -- A list of word vectors from negative words used to train the classifier.
                        They are expected to come from the same embedding model given as
                        `word_vector_getter`. This can be used as an alternative to the
                        `negative_words` to speed up the calculation if the class is re-generated
                        multiple times. Also, this can only be used together with `positive_vectors`
                        and not with `positive_words`.
    word_vector_getter -- An object that returns a vector given a word as parameter to the
                          `__getitem__()` function.
    random_state -- The random state to be used for shuffling the data before returning it.
    """

    def __init__(
            self,
            word_vector_getter,
            positive_words: list = None,
            negative_words: list = None,
            positive_vectors: list = None,
            negative_vectors: list = None,
            random_state: int = 42):

        self.word_vector_getter = word_vector_getter
        self.random_state = random_state

        # If words are provided, embedd them first
        if positive_words and negative_words:
            positive_vectors, negative_vectors = self._get_sentiment_vectors(
                positive_words, negative_words)

        # Load words lists from disk, embed them and transform them into a labled tuple
        self.X, self.y = self._prepare_sentiment_data(positive_vectors, negative_vectors)

        # Retrieve the trained logistic regression classifier
        self.lrc = self._train_lrc_sentiment_classifier()

    def _get_sentiment_vectors(self, positive_words: list, negative_words: list) -> tuple:
        """Retrieve the embedding vectors for the provided words from the provided model.

        Return a tuple of positive word vectors and negative word vectors.

        Arguments:
        positive_words -- A list of positive words used to train the classifier.
        negative_words -- A list of negative words used to train the classifier."""
        # Receive embeddings for the sentiment words
        # Words for which no vector exists are skipped and not included in the returned set
        positive_vectors = []
        negative_vectors = []

        for token in positive_words:
            try:
                positive_vectors.append(self.word_vector_getter[token])
            except KeyError:
                logging.debug(f"Couldn't find vector for token {token}. Skipping.")

        for token in negative_words:
            try:
                negative_vectors.append(self.word_vector_getter[token])
            except KeyError:
                logging.debug(f"Couldn't find vector for token {token}. Skipping.")

        return positive_vectors, negative_vectors

    def _prepare_sentiment_data(self, positive_vectors: list, negative_vectors: list) -> tuple:
        """Prepare sentiment data for classification.

        Return it as labeled data in the form of X and y.

        Arguments:
        positive_vectors -- A list of word vectors from positive words used to train the classifier.
        negative_vectors -- A list of word vectors from negative words used to train the classifier.
        """
        # Preapre data in X, y format; shuffle the data before returning
        # Positive vectors will have a 0 label, negative a 1 label
        X = [*positive_vectors, *negative_vectors]
        y = [*np.zeros(len(positive_vectors)), *np.ones(len(negative_vectors))]

        return shuffle(X, y, random_state=self.random_state)

    def _train_lrc_sentiment_classifier(self):
        """Train a Logistic Regression classifier on the sentiment terms from [1].

        Return the trained classifier object.

        [1] https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
        """
        # Initialize and fir the LRC using sklearn
        lrc = LogisticRegression(random_state=self.random_state)
        lrc.fit(self.X, self.y)

        return lrc

    def _kullback_leibler_divergence(self, distribution_i: list, distribution_j: list) -> np.array:
        """Calculate the difference between two distributions using the Kullback-Leibler divergence.

        Return the result of the calculation as a list of divergences. The implemented algorithm is
        presented in [1] and can be formulated as:
        $D_{KL}(p,q) = \sum_{i=1}^N p_i * log(\frac{p_i}{q_i})$.

        [1] https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained

        Arguments:
        distribution_i -- The source distribution.
        distribution_j -- The destination distribution to which compare the `distribution_i` to.
        """
        if len(distribution_i) != len(distribution_j):
            raise ValueError("Both distributions need to be of the same length.")

        divergences = [
            (p * np.log(p / distribution_j[idx])) for idx, p in enumerate(distribution_i)]

        return np.array(divergences)

    def get_score(self, identity_terms: list) -> tuple:
        """Calculate the RNSB score for the given identity terms.

        Returns the result of the RNSB calculation presented in [1], alognside the normalized
        identity distributions and a list of OOV tokens for the given embedding. The former is
        basically the sum of the Kullback-Leibler divergence of the normalized differences between
        the predicted negative sentiments of the identity terms and a uniform distribution thereof.
        It can be formulated as: $RNSB(P) = D_{KL}(P||U)$, where $P$ is a set of normalized
        probabilities of negative sentiments for all identity terms and $U$ is a uniform
        distribution.

        To predict the negative sentiments, a logistic regression classifier is trained on word
        vectors of sentiment words from [2] using the given vector space, as originally proposed
        by [1].

        [1] dx.doi.org/10.18653/v1/P19-1162
        [2] https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

        Arguments:
        identity_terms -- A list of terms that describe different groups of the property that
                          should be investigated, such as national origin identity. In [1] those
                          properties are also referred to as protected groups. If this is a list of
                          lists, the inner lists are expected to be vectors.
        """
        # Try to embed all identity terms; ignore them if not in embdding vocabulary
        # Log missing words to console
        identity_vectors = []
        oov_tokens = []
        if type(identity_terms[0]) == str:
            for term in identity_terms:
                try:
                    identity_vectors.append(self.word_vector_getter[term])
                except KeyError:
                    logging.debug(f"Term '{term}' is OOV. Ignoring.")
                    oov_tokens.append(term)
        elif type(identity_terms[0]) == list or type(identity_terms[0]) == np.ndarray:
            identity_vectors = identity_terms

        # Predict the negative probabilities of the identity terms/vectors
        # According to read_sentiment_data, negative probailities are at position 1
        identity_probabilities = [probas[1] for probas in self.lrc.predict_proba(identity_vectors)]

        # Calculate normalized probabilities to be able to handle them as distribution
        identity_probas_normalized = [
            (proba / sum(identity_probabilities)) for proba in identity_probabilities]

        # Create a uniform distribution of the probabilities and calculate the final score
        uniform_distribution = np.full(
            len(identity_probas_normalized), 1 / len(identity_probas_normalized))

        result = np.sum(
            self._kullback_leibler_divergence(identity_probas_normalized, uniform_distribution))

        return (
            result,
            identity_probas_normalized,
            oov_tokens)


def rnsb_evaluation(
        embedding_model,
        lexicon: dict) -> dict:
    """Evaluate the RNSB metric with the given lexicon.

    Return a dict containing the results of the different shuffled runs per test type.
    Each index in the type list represents the results for one shuffled lexicon (or a shuffled
    lexicon combination if there are two lexicons) in the form of a list. In that list, each index
    `i` represents a lexicon of size `i * m` where `m` is the step size. Thus, it also represents
    the `start_size` ofthe lexicon. Further, `i` also represents the random state used to shuffle
    the original list.

    Not that, since the RNSB test uses only a single vector for each social group, this evaluation
    averages the words in the target group sets. This average is then used to represent the group.

    Arguments:
    embedding_model -- The embedding model to use for the evaluation.
    lexicon -- The lexicon to be used for the evaluation. Is expected to have the following keys:
               target_set_1, target_set_2, attribute_set_1, attribute_set_2.
    """
    # Dict that holds the results
    rnsb_results = {}

    # For each bias type (e.g. gender, ethnicity, religion) ...
    for test_type, test_lexicons in lexicon.items():
        test_type_results = {
            "shuffled_attribute_results": [],
            "shuffled_target_results": [],
            "attribute_set_lengths": [],
            "target_set_lengths": []}

        # --------------------------------------------------------------------------------
        # Conduct evaluation with shuffled attribute lists

        word_vector_cache = build_word_embedding_cache(lexicon, embedding_model)

        # Averaging the target sets before the shuffle loop saves some time for the shuffled
        # attribute set runs (as the targets will stay the same)
        averaged_target_sets = [
            np.mean([embedding_model[t] for t in test_lexicons["target_set_1"][0][-1]], axis=0),
            np.mean([embedding_model[t] for t in test_lexicons["target_set_2"][0][-1]], axis=0)]

        # Combine the two attribute sets to always use the same index element from both
        shuffled_attribute_runs = list(zip(
            test_lexicons["attribute_set_1"], test_lexicons["attribute_set_2"]))

        attribute_progress_bar = tqdm(shuffled_attribute_runs, desc=f"RNSB-{test_type}-attributes")
        for shuffled_run in attribute_progress_bar:
            shuffled_attribute_run_results = []

            partial_progress_bar = tqdm(list(zip(shuffled_run[0], shuffled_run[1])), leave=False)
            for partial_attribute_1, partial_attribute_2 in partial_progress_bar:
                result = rnsb = RNSB(
                    word_vector_cache,
                    positive_words=partial_attribute_1,
                    negative_words=partial_attribute_2)
                shuffled_attribute_run_results.append(rnsb.get_score(averaged_target_sets)[0])

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

        # Initialzing an RNSB instance before the shuffle loop saves some time for the shuffled
        # target set runs (the rnsb instance will stay the same as the attribute sets wont change)
        rnsb_all_attribute_terms = RNSB(
            embedding_model,
            positive_words=test_lexicons["attribute_set_1"][0][-1],
            negative_words=test_lexicons["attribute_set_2"][0][-1])

        # Combine the two target sets to always use the same index element from both
        shuffled_target_runs = list(zip(
            test_lexicons["target_set_1"], test_lexicons["target_set_2"]))

        target_progress_bar = tqdm(shuffled_target_runs, desc=f"RNSB-{test_type}-targets")
        for shuffled_run in target_progress_bar:
            shuffled_target_run_results = []

            partial_progress_bar = tqdm(list(zip(shuffled_run[0], shuffled_run[1])), leave=False)
            for partial_target_1, partial_target_2 in partial_progress_bar:
                averaged_shuffled_target_sets = [
                    np.mean([embedding_model[t] for t in partial_target_1], axis=0),
                    np.mean([embedding_model[t] for t in partial_target_2], axis=0)]

                result = rnsb_all_attribute_terms.get_score(averaged_shuffled_target_sets)
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

        rnsb_results[test_type] = test_type_results

    return rnsb_results
