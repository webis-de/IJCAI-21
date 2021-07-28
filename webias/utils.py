####################################################################################################
#
# Utility functions that might be useful across different files and scripts.
#
####################################################################################################

from itertools import zip_longest
from sklearn.utils import shuffle


def print_separator(text: str) -> None:
    """Print a nice separator to the console with the given text.

    Arguments:
    text -- The texts that should be printed under the separator.
    """
    print("")
    print(f"\033[91m{'=' * 80}\033[00m")
    print(f"\033[91m= > {text}\033[00m")


def extract_unique_tokens(lexicon: dict) -> list:
    """Extract all unique tokens from a lexicon and return it as list.

    The lexicon is expected to have the format returned by the `prepare_lexicons` function.

    Arguments:
    lexicon -- A dictionary containing all the tokens that should be extracted into a unique list of
               tokens.
    """
    all_tokens = []

    # Collect all possible tokens from the provided lexicon
    for test_type, test_lexicons in lexicon.items():
        for words in test_lexicons.values():
            all_tokens.extend(words[0][-1])

    # Remove duplicates and return as list
    return list(set(all_tokens))


def build_word_embedding_cache(lexicon: dict, embedding_model) -> dict:
    """Retrieve the word vectors for all tokens in the provided lexicons and cache them.

    Return the cache as a dictionary.
    This should decrease the access times in cases where the vectors are requested multiple hundred
    times. Since the runs should all have the same tokens, use the last index of the first run and
    extract all tokens.

    Arguments:
    lexicon -- A dictionary containing the tokens that should be cached. Expects the lexcion to be
               in a specific format, as returned by the lexicon preparation function
               `prepare_lexicons`.
    embedding_model -- The model that should be used to retrieve the word vectors.
    """
    unique_tokens = extract_unique_tokens(lexicon)

    # Save embeddings for all tokens in the lexicon as a dict
    return {key: embedding_model[key] for key in unique_tokens}


def _determine_combined_lexicon_eval_lengths(
        lexicons: list,
        step_size: int,
        allow_different_lengths: bool) -> list:
    """Determine the lengths at which the given lexicons should be evaluated at.

    Return a list of lexicon sizes.

    The lexicon sizes will start at `step_size` and end at the combined length of all given
    lexicons.

    Arguments:
    lexicons -- The lexcions for which the evlauation lengths should be determined.
    step_size -- The number of words to be added to each lexicon at each step.
    allow_different_lengths -- Whether the evaluation allows for differently sized lexicon lengths
                               to be used or not.
    """
    lexicon_lengths = [len(lex) for lex in lexicons]

    # If different lengths are allowed, the maximum lexicon size will be the combination of all
    # words in all lexicons and thus the sum of all lexicons lengths. Otherwise, we need to use the
    # length of the smallest lexicon and cut all other lexicons to its size and thus have a maximum
    # length of min-length * #lexicons.
    if allow_different_lengths:
        max_lexicon_size = sum(lexicon_lengths)
    else:
        max_lexicon_size = min(lexicon_lengths) * len(lexicons)

    # Since we are combining all given lexicons later on, we need to increase the step size here to
    # be the product of the #lexicons with the provided step size.
    combined_step_size = step_size * len(lexicons)

    lexicon_eval_lengths = list(range(combined_step_size, max_lexicon_size + 1, combined_step_size))

    # In cases where the division of the maximum length by the step size does leave a rest we need
    # to add the last lexicon length step manually; we would miss it otherwise
    if max_lexicon_size % combined_step_size > 0:
        lexicon_eval_lengths.append(max_lexicon_size)

    return lexicon_eval_lengths


def _determine_lexicon_eval_lengths(
        lexicons: list,
        step_size: int,
        allow_different_lengths: bool) -> list:
    """Determine the lengths at which the given lexicons should be evaluated at.

    Return a list of lexicon sizes.

    The lexicon sizes will start at `step_size` and end at the length of the longest lexicon if
    different lengths are allowed and the length of the shortest one otherwise.

    Arguments:
    lexicons -- The lexcions for which the evlauation lengths should be determined.
    step_size -- The number of words to be added to each lexicon at each step.
    allow_different_lengths -- Whether the evaluation allows for differently sized lexicon lengths
                               to be used or not.
    """
    lexicon_lengths = [len(lex) for lex in lexicons]

    # If different lengths are allowed, the maximum lexicon size will be the size of the longest
    # lexicon. Otherwise, we need to use the length of the smallest lexicon and cut all other
    # lexicons to its size.
    if allow_different_lengths:
        max_lexicon_size = max(lexicon_lengths)
    else:
        max_lexicon_size = min(lexicon_lengths)

    lexicon_eval_lengths = list(range(step_size, max_lexicon_size + 1, step_size))

    # In cases where the division of the total length by the step size does leave a rest we need to
    # add the last lexicon length step manually; we would miss it otherwise
    if max_lexicon_size % step_size > 0:
        lexicon_eval_lengths.append(max_lexicon_size)

    return lexicon_eval_lengths


def prepare_combined_lexicons(
        lexicon_1: list,
        lexicon_2: list,
        shuffled_runs: int,
        step_size: int,
        lowercase: bool,
        allow_different_lengths: bool = False) -> list:
    """Combine and prepare a two lists of tokens for the metric evaluation.

    Return a list of shuffled runs for a lexicon that is created by combining all given ones. This
    method ensures that each partial lexicon (parts of the full lexicons at different runs) will
    contain roughly the same number of tokens from each of the lexicons.

    Arguments:
    lexicon_1 -- The first list of tokens that should be prepared.
    lexicon_2 -- The second list of tokens that should be prepared.
    shuffled_runs -- How many shuffled lexcions to prepare. Each run will have a different shuffle.
    step_size -- By which size to increase the lexicons for each test.
    lowercase -- Whether the tokens should be lowercased or not.
    allow_different_lengths -- Whether to allow for different lexicon lengths or not. If `False`,
                               all lexicons will be trimed to the size of the shortest one.
    """
    # Lowercase the provided tokens, if required
    prepared_lexicons = [[
        t.lower() for t in inner]
        for inner in [lexicon_1, lexicon_2]] if lowercase else [lexicon_1, lexicon_2]
    shortest_lexicon = min([len(lex) for lex in prepared_lexicons])

    lexicon_lengths = _determine_combined_lexicon_eval_lengths(
        prepared_lexicons, step_size, allow_different_lengths)

    # All runs of the final lexicon
    lexicon_runs = []

    for run in range(0, shuffled_runs):
        # All partial lexicons of the current run (including the full lexicon at the end)
        shuffled_partials = []

        if allow_different_lengths:
            # Zip all lexicons into tuples and flatten the list of tuples afterwards.
            # For shorter lists, the later tuples contain "None" as value, which will basically
            # lead to a final, flattened list where the later tokens are from the longer lexicon
            # exclusively (before shuffling that is).
            lexicons_zip = zip_longest(*prepared_lexicons, fillvalue=None)
            lexicons_unpkg = [
                t for t_tuple in lexicons_zip for t in t_tuple if t is not None]

            # Shuffle the combined and flattened lexicons
            shuffled_lexicons_unpkg = shuffle(lexicons_unpkg, random_state=run)
        else:
            # Trim each of the given lexicons to the size of the shortest lexicon and shuffle
            # them individually
            shuffled_lexicons = [
                shuffle(lexicon[:shortest_lexicon], random_state=run)
                for lexicon in prepared_lexicons]

            # Zip all shuffled lexicons into tuples and flatted the list of tuples afterwards
            shuffled_lexicons_zip = zip(*shuffled_lexicons)
            shuffled_lexicons_unpkg = [t for t_tuple in shuffled_lexicons_zip for t in t_tuple]

        # Split the final shuffled and combined lexicon into multiple partials
        for length in lexicon_lengths:
            shuffled_partials.append(shuffled_lexicons_unpkg[:length])

        lexicon_runs.append(shuffled_partials)

    # Return a the runs for a single lexicon
    return lexicon_runs


def prepare_lexicons(
        lexicon_1: list,
        lexicon_2: list,
        shuffled_runs: int,
        step_size: int,
        lowercase: bool,
        allow_different_lengths: bool = False) -> list:
    """Prepare a two lists of tokens for the metric evaluation.

    Return a tuple of all given lexicons that were shuffled and split separately.

    Arguments:
    lexicon_1 -- The first list of tokens that should be prepared.
    lexicon_2 -- The second list of tokens that should be prepared.
    shuffled_runs -- How many shuffled lexcions to prepare. Each run will have a different shuffle.
    step_size -- By which size to increase the lexicons for each test.
    lowercase -- Whether the tokens should be lowercased or not.
    allow_different_lengths -- Whether to allow for different lexicon lengths or not. If `False`,
                               all lexicons will be trimed to the size of the shortest one. If
                               `True` the sizes of the lists will be increased proportional to their
                               relative length to each other. The shortest list will use
                               the defined step size, while the longer lists will use a step size
                               that makes them grow proportionally, so that the relative length
                               difference is the same for all test runs.
    """
    # Lowercase the provided tokens, if required
    prepared_lexicons = [[
        t.lower() for t in inner]
        for inner in [lexicon_1, lexicon_2]] if lowercase else [lexicon_1, lexicon_2]

    lexicon_lengths = _determine_lexicon_eval_lengths(
        prepared_lexicons, step_size, allow_different_lengths)

    # List of all lexicons, where each element holds the runs for one of them
    all_lexicon_runs = []

    for lexicon in prepared_lexicons:
        # All runs of the current lexicon
        lexicon_runs = []

        for run in range(0, shuffled_runs):
            # All partial lexicons of the current run (including the full lexicon at the end)
            shuffled_partials = []

            # Shuffle first, then split the shuffled lexicon into multiple partials
            # If different lengths are not allowed, we need to additionally trim the lexicons
            # to the length of the shortest before shuffling to make sure that all runs contain
            # the same vocabulary.
            if allow_different_lengths:
                shuffled_lexicon = shuffle(lexicon, random_state=run)
            else:
                shuffled_lexicon = shuffle(lexicon[:lexicon_lengths[-1]], random_state=run)
            # Due to the working of the list indexing `:length`, this will also ensure that each
            # run has always the same number of steps. In case of a shorter lexicon, the
            # `:length` will just "overshoot" it at add the full lexicon to the list again
            # (which is what we want). In cases where we want the lexicons to have the same size
            # we need to take care of trimming them to that size above and this part still works
            # the same.
            for length in lexicon_lengths:
                shuffled_partials.append(shuffled_lexicon[:length])

            lexicon_runs.append(shuffled_partials)

        all_lexicon_runs.append(lexicon_runs)

    # Return a tuple of all lexicons, each containing the runs for each lexicon
    return tuple(all_lexicon_runs)
