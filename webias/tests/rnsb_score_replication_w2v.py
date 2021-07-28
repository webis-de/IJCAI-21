import json
import unittest

from os import path

from ..constants import RNSB_TEST_TOLERANCE
from ..word_vectors import WordVectors
from ..rnsb import RNSB


class TestRnsbWord2Vec(unittest.TestCase):
    # Load test data from file
    rnsb_tests = path.join(path.dirname(path.dirname(__file__)), "data", "rnsb_tests.json")
    with open(rnsb_tests, "r") as f:
        rnsb_tests = json.load(f)

    # Load word vectors
    vectors = WordVectors("word2vec")

    # Read positive/negative words
    with open(path.join("..", "data", "positive-words.txt"), "r") as f:
        positive_words = f.read().split("\n")[31:]
    with open(path.join("..", "data", "negative-words.txt"), "r") as f:
        negative_words = f.read().split("\n")[31:]

    def test_one(self):
        # Initialize RNSB metric
        rnsb = RNSB(
            self.__class__.vectors,
            positive_words=self.__class__.positive_words,
            negative_words=self.__class__.negative_words)

        # Retrieve test data
        test_data = self.__class__.rnsb_tests["test1"]

        self.assertAlmostEqual(
            rnsb.get_score(test_data["identity_words"])[0],
            test_data["word2vec_result"],
            delta=RNSB_TEST_TOLERANCE)
