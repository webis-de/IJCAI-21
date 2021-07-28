import argparse
import json
import logging
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from os import path

from webias.constants import BIAS_METRIC_LIMITS, COLORS, LOGGING_CONFIG


def plot_results(
        model_names: tuple,
        model_data: tuple,
        y_limits: tuple,
        metric_name: str,
        output_filename: str,
        supplemental_plot_info: list = [],
        title: str = "(Social) bias metric evaluation",
        colors: tuple = ("red", "blue"),
        plot_individual_runs: bool = False):
    """Create a plot for the given model data that compares bias scores for specific metric.

    Arguments:
    model_names -- Tuple containing the names of the models that should be plotted. Names will
                   always be cut to 15 characters.
    model_data -- Tuple of dicts, each containing the data necessary for the plot. A dict must
                  contain at least the following keys, each providing access to a list of points:
                  "runs" (list of lists), "mean_values" (list), "min_values" (list), "max_values"
                  (list), "x_values" (list)
    y_limits -- Tuple describing the explicit limits of the y-axis in the form of (min, max).
    metric_name -- Name of the metric that is being evaluated.
    output_filename -- The full path (incl. filename) of the file to which the plot should be saved.
    supplemental_plot_info -- A list string that should be added to the bottom of the plot legend.
    title -- Title of the plot.
    colors -- A list of colors that should be used for the plots.
    plot_individual_runs -- Whether to plot individual runs inside the silhouette or not.
    """

    # Initialize the plot
    fig, ax = plt.subplots()

    # Set explicit limits to the y-axis to ensure comparability with other plots of the same metric
    ax.set_ylim(ymin=y_limits[0], ymax=y_limits[1])

    # Collecting the min and max y-values to add them to the legend below
    max_y_values = []
    min_y_values = []

    # For each of the provided models
    for index, data in enumerate(model_data):
        color = colors[index]
        model_name = model_names[index][:15]
        model_label = (
            f"{model_name} "
            f"({round(data['robustness_score'], 4)})")
        x_values = data["x_values"]

        max_y_values.extend(data["max_values"])
        min_y_values.extend(data["min_values"])

        # Plot original/single results
        if plot_individual_runs:
            for run in data["runs"]:
                ax.plot(x_values, run, color=COLORS[color]["medium"], linewidth=0.5)

        # Plot mean values
        ax.plot(
            x_values,
            data["mean_values"],
            linestyle="dashed",
            linewidth=0.5,
            color=COLORS[color]["dark"])

        # Plot min values
        ax.plot(x_values, data["min_values"], linewidth=0.5, color=COLORS[color]["dark"])

        # Plot max values
        ax.plot(
            x_values,
            data["max_values"],
            linewidth=0.5,
            color=COLORS[color]["dark"],
            label=model_label)

        # Fill silhouette area between min and max lines
        ax.fill_between(
            x_values,
            data["min_values"],
            data["max_values"],
            color=COLORS[color]["light"])

    # Explicitly set min and max values for x-axis
    ax.set_xlim(left=0, right=max(x_values))

    # Set axes labels and title
    ax.set_xlabel(f"Combined number of words in {args.shuffle_type} wordlist(s)")
    ax.set_ylabel(f"{metric_name.upper()} value")
    ax.set_title(title)

    # Add min and max values of both axes to the supplemental information
    x_min_max = f"Xmin: {min(x_values)} ; Xmax: {max(x_values)}"
    y_min_max = f"Ymin: {round(min(min_y_values), 4)} ; Ymax: {round(max(max_y_values), 4)}"
    supplemental_plot_info.extend([x_min_max, y_min_max])

    # Add supplemental information to the graph legend
    handles, labels = ax.get_legend_handles_labels()
    for info in supplemental_plot_info:
        handles.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
        labels.append(info)
    ax.legend(handles, labels)

    # Calculate and set figure width
    fig_width_cm = 20
    # 3:2 ratio identified by max width; multiplied by cm-to-inch factor
    fig.set_size_inches((fig_width_cm * 0.394, (fig_width_cm / 3 * 2) * 0.394))

    # Save plot to disk
    logging.info(f"Saving plot '{title}' to disk at '{output_filename}'.")
    plt.savefig(fname=output_filename, dpi=300)


def main():
    # Read the provided evaluation results from disk
    logging.info("Reading evlauation results from provided file.")
    with open(args.results_file, "r") as f:
        evaluation_results = json.load(f)

    # For each of the metrics specified in the command line
    for metric_name in args.metrics:
        model_names = []
        model_data = []

        # ----------------------------------------------------------------------
        # Accumulate all necessary data points
        accuracy_score = 0.0

        # For each evaluated model in the file, accumulate the evaluation data
        for model_name, data in evaluation_results.items():
            results_key = f"shuffled_{args.shuffle_type}_results"

            # If the current key is actually not a model
            if model_name == "accuracy_score":
                accuracy_score = data[metric_name][args.test_type][results_key]
                continue
            model_names.append(model_name)

            # Read the original results file
            og_results_dir = args.original_results_dir if args.original_results_dir else args.output
            original_results_file = path.join(og_results_dir, data["original_results_file"])
            with open(original_results_file, "r") as f:
                original_results = json.load(f)

            model_statistics = data["statistics"][metric_name][args.test_type][results_key]
            original_test_type_results = original_results[metric_name][args.test_type]

            # Collect all information in a dictionary and a format that the plotting
            # function requires
            model_data.append({
                "runs": original_test_type_results[results_key],
                "mean_values": model_statistics["mean_values"],
                "min_values": model_statistics["min_values"],
                "max_values": model_statistics["max_values"],
                "silhouette_size": model_statistics["silhouette_size"],
                "graph_coverage": model_statistics["graph_coverage"],
                "robustness_score": model_statistics["robustness_score"],
                "x_values": original_test_type_results[f"{args.shuffle_type}_set_lengths"]})

        # ----------------------------------------------------------------------
        # Generate a plot from the accumulated data points
        plot_title = f"{metric_name.upper()}--{args.test_type}--shuffled_{args.shuffle_type}"
        filename = path.join(args.output, f"{plot_title}.svg")
        plot_results(
            model_names=model_names,
            model_data=model_data,
            y_limits=BIAS_METRIC_LIMITS[metric_name],
            output_filename=filename,
            metric_name=metric_name,
            supplemental_plot_info=[
                f"Accuracy score: {round(accuracy_score, 4)}"],
            colors=args.plot_colors,
            title=plot_title,
            plot_individual_runs=args.plot_individual_runs)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to plot the bias metric evaluation results.")

    parser.add_argument(
        "-i",
        "--results_file",
        required=True,
        type=str,
        help="Path to the file containing the evaluation results that should be plotted.",
        metavar="RESULTS_FILE")
    parser.add_argument(
        "-d",
        "--original_results_dir",
        default=None,
        type=str,
        help="Path to the directory containing the original result files. If `None`, it is assumed "
             "that the files are in the output directory.",
        metavar="ORIGINAL_RESULTS_DIR")
    parser.add_argument(
        "-m",
        "--metrics",
        required=True,
        nargs="+",
        type=str,
        help="A list of metrics for which the evlauation results should be plotted (whitespace "
             "separated).",
        metavar="METRICS")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the directory where the plot files should be written to.",
        metavar="OUTPUT_DIR")
    parser.add_argument(
        "-c",
        "--plot_colors",
        choices=list(COLORS.keys()),
        default=["red", "blue"],
        nargs="+",
        type=str,
        help="The colors that should be used to plot the models. Note that the former color will "
             "always be used for the first model in the given results file.",
        metavar="PLOT_COLORS")
    parser.add_argument(
        "-s",
        "--shuffle_type",
        required=True,
        type=str,
        help="Defines the type of word set (e.g. attribute or target) for which the data should be "
             "should be plotted.",
        metavar="SHUFFLE_TYPE")
    parser.add_argument(
        "-t",
        "--test_type",
        required=True,
        type=str,
        help="Defines the type of test (e.g. gender or ethnicity) for which the data should be "
             "plotted.",
        metavar="TEST_TYPE")
    parser.add_argument(
        "-r",
        "--plot_individual_runs",
        action="store_true",
        help="Add this flag if lines for the individual runs should be added inside the "
             "silhouettes. Note: this can significantly increase the file size of the plot, "
             "depending on the number of runs conducted.")

    args = parser.parse_args()
    logging.basicConfig(**LOGGING_CONFIG)

    main()
    logging.info("Done.")
