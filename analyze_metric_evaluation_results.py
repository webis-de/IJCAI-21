import argparse
import json
import logging
import numpy as np

from datetime import datetime
from os import path

from webias.constants import BIAS_METRIC_LIMITS, BIAS_METRIC_ZERO, LOGGING_CONFIG


def calculate_mean_values(results_per_run: list) -> float:
    """Calculate the mean values of each dimension for the given lists of runs.

    Return a list of means.

    Arguments:
    results_per_run: A list of lists, where each inner list contains evaluation results per run.
    """
    results_by_list_length = list(zip(*results_per_run))

    return np.mean(results_by_list_length, axis=1).tolist()


def calculate_max_values(results_per_run: list) -> float:
    """Calculate the maximum values of each dimension for the given lists of runs.

    Return a list of maximum values.

    Arguments:
    results_per_run: A list of lists, where each inner list contains evaluation results per run.
    """
    results_by_list_length = list(zip(*results_per_run))

    return np.max(results_by_list_length, axis=1).tolist()


def calculate_min_values(results_per_run: list) -> float:
    """Calculate the minimum values of each dimension for the given lists of runs.

    Return a list of minimum values.

    Arguments:
    results_per_run: A list of lists, where each inner list contains evaluation results per run.
    """
    results_by_list_length = list(zip(*results_per_run))

    return np.min(results_by_list_length, axis=1).tolist()


def calculate_silhouette_area(lower_bounds: list, upper_bounds: list, x_values: list) -> float:
    """Calculate the silhouette area between two curves that are defined by the given bounds.

    Return the size of the area.

    Uses the numpy implementation of the trapezoidal rule to approximate the area between the two
    curves.

    Arguments:
    lower_bounds -- The points defining the graph of the lower bounds of the silhouette.
    upper_bounds -- The points defining the graph of the upper bounds of the silhouette.
    x_values -- The points on the x-axis at which the lower and upper bounds were measured.
    """
    upper_graph_area = np.trapz(y=upper_bounds, x=x_values)
    lower_graph_area = np.trapz(y=lower_bounds, x=x_values)

    return upper_graph_area - lower_graph_area


def calculate_graph_coverage(
        metric: str,
        x_axis_size: int,
        silhouette_size: float,
        min_y_value: int = None,
        max_y_value: int = None) -> float:
    """Calculate the percentage of the available area covered by the given silhouette.

    Return the percentage covered.

    Arguments:
    metric -- The metric that was used. Important to figure out upper and lower y-value bounds.
    x_axis_size -- The largest x-value of the measurement.
    silhouette_size -- The size of the area covered by the silhouette.
    min_y_value -- The minimum value on the y-axis. Necessary in case the metric is unknown or not
                   applicable.
    max_y_value -- The maximum value on the y-axis. Necessary in case the metric is unknown or not
                   applicable.
    """
    if min_y_value and max_y_value:
        y_axis_length = max_y_value - min_y_value
    else:
        y_axis_length = BIAS_METRIC_LIMITS[metric][1] - BIAS_METRIC_LIMITS[metric][0]

    return silhouette_size / (x_axis_size * y_axis_length)


def calculate_model_statistics(evaluation_results: dict) -> dict:
    """Calculate different statistics for the given evaluation data, such as graph coverage.

    Return a dictionary containing the different statistical results for each evaluation in the
    given results file.

    Arguments:
    evaluation_results -- Dictionary containing the results of a specific model for different
                          metrics. Each top-level key is expected to be a metric name. An exception
                          is the top-level key "model_name", which is expected to hold an identifier
                          for the evaluated model.
    """
    model_analysis_results = {}

    # For each metric evaluation of the evaluated model...
    for metric_name, results_by_type in evaluation_results.items():
        # If the current item is not actually a metric evaluation
        if metric_name == "model_name":
            continue

        model_analysis_results[metric_name] = {}

        # For each test type of the current metric
        for test_type, results in results_by_type.items():
            model_analysis_results[metric_name][test_type] = {}

            # For each shuffle type...
            for shuffle_type, values in results.items():
                # If the current item is not actually a list of values
                if shuffle_type == "attribute_set_lengths" or shuffle_type == "target_set_lengths":
                    continue

                shuffled_list = shuffle_type.split("_")[1]
                set_lengths = results[f"{shuffled_list}_set_lengths"]

                mean_values = calculate_mean_values(values)
                max_values = calculate_max_values(values)
                min_values = calculate_min_values(values)
                silhouette_size = calculate_silhouette_area(
                    min_values, max_values, set_lengths)
                graph_coverage = calculate_graph_coverage(
                    metric_name,
                    max(set_lengths),
                    silhouette_size)
                robustness_score = get_robustness_score(graph_coverage)

                statistics = {
                    "mean_values": mean_values,
                    "max_values": max_values,
                    "min_values": min_values,
                    "silhouette_size": silhouette_size,
                    "graph_coverage": graph_coverage,
                    "robustness_score": robustness_score,
                    "x_values": set_lengths}

                model_analysis_results[metric_name][test_type][shuffle_type] = statistics

    return model_analysis_results


def calculate_accuracy_scores(model_statistics: dict) -> dict:
    """Calcuate the area between each two mean curves in the given data.

    Return a dict with graph coverage (calculated area normalized by the total graph size) for each
    of the metrics in the given data.

    Arguments:
    model_statistics -- Dictionary holding the model statistics data. Each top-level key is
                        expected to be a model identifier, with the value being the respective
                        model statistics.
    """
    accuracy_scores = {}

    # Extract actual statistic lists from given dict
    model_statistics_unpkg = [
        model_statistics[model]["statistics"] for model in model_statistics.keys()]

    # For each metric evaluation of the evaluated model...
    # (here we can always just use the keys of the first models as they are supposed to be equal
    # for both models; otherwise this analysis wont work anyway)
    for metric_name in model_statistics_unpkg[0].keys():
        accuracy_scores[metric_name] = {}

        # For each test type of the current metric...
        for test_type in model_statistics_unpkg[0][metric_name].keys():
            accuracy_scores[metric_name][test_type] = {}

            # For each shuffle type...
            for shuffle_type in model_statistics_unpkg[0][metric_name][test_type].keys():
                mean_values_model_1 = np.array(
                    model_statistics_unpkg[0][metric_name][test_type][shuffle_type]["mean_values"])
                mean_values_model_2 = np.array(
                    model_statistics_unpkg[1][metric_name][test_type][shuffle_type]["mean_values"])
                x_values = np.array(
                    model_statistics_unpkg[0][metric_name][test_type][shuffle_type]["x_values"])

                # Get the absolute values of the curves to bring them both into the same range
                # Necessary for measures where bias can also be negative, like WEAT
                abs_means_1 = np.abs(mean_values_model_1)
                abs_means_2 = np.abs(mean_values_model_2)
                area_between_means = calculate_silhouette_area(abs_means_2, abs_means_1, x_values)

                # Calculate graph coverage
                coverage = calculate_graph_coverage(
                    metric_name,
                    max(x_values),
                    area_between_means,
                    min_y_value=BIAS_METRIC_ZERO[metric_name],
                    max_y_value=BIAS_METRIC_LIMITS[metric_name][1])

                accuracy_scores[metric_name][test_type][shuffle_type] = get_accuracy_score(coverage)

    return accuracy_scores


def get_accuracy_score(mean_silhouette_coverage: float) -> float:
    """Calculate the accuracy score given the coverage of the graph by the silhouette between means.

    Return the accuracy score.

    Arguments:
    mean_silhouette_coverage -- Coverage of the total graph by the respective silhouette between
                                means.
    """
    return 0.5 + (0.5 * mean_silhouette_coverage)


def get_robustness_score(silhouette_coverage: float) -> float:
    """Calculate the robustness score given the coverage of the graph by the silhouette.

    Return the robustness score.

    Arguments:
    silhouette_coverage -- Coverage of the total graph by the respective silhouette.
    """
    return 1 - silhouette_coverage


def main():
    # Statistics calculated for each model independently of the other
    intra_model_statistics = {}

    # For each model evaluation file...
    for model_results_file in args.evaluation_results:
        # Load embedding model evlauation results from disk
        logging.info("Loading results file from disk.")
        with open(model_results_file, "r") as f:
            evaluation_results = json.load(f)

        model_name = evaluation_results["model_name"]
        model_statistics = calculate_model_statistics(evaluation_results)

        logging.info("Calculating intra model statistics.")
        intra_model_statistics[model_name] = {
            "original_results_file": path.basename(model_results_file),
            "statistics": model_statistics}

    # Statistics calculated depending on both models
    logging.info("Calculating inter model statistics.")
    inter_model_statistics = {
        "accuracy_score": calculate_accuracy_scores(intra_model_statistics)}

    # Write results to disc
    dt = datetime.today().strftime("%Y%m%d%H%M%S")
    filename = args.output_filename if args.output_filename else f"evaluation-analysis-{dt}.json"
    output_file = path.join(args.output, filename)

    logging.info(f"Writing results to disk at '{output_file}'.")

    with open(output_file, "w") as f:
        final_results = {
            **inter_model_statistics,
            **intra_model_statistics}
        json.dump(final_results, f)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to analyze previously calculated evaluation results and prepare them for "
        "plotting.")

    parser.add_argument(
        "-r",
        "--evaluation_results",
        required=True,
        nargs="+",
        type=str,
        help="List of paths to the evaluation results files.",
        metavar="EVALUATION_RESULTS")
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

    args = parser.parse_args()
    logging.basicConfig(**LOGGING_CONFIG)

    main()
    logging.info("Done.")
