#! /bin/bash

# ----------PARAMETER DEFINITIONS----------------------------------------

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
# All word embeddings models that should be used to evaluate the metrics.
# If the embedding model is supported by the embeddings library, simply
# add the name of the model in both values. If the model is loaded from a
# custom word2vec-format file, use the following syntax:
# `["numberbatch1908"]="data/word-vectors/numberbatch-en-1908.txt"`
declare -A MODEL_PATHS=(\
    ["glove"]="glove" \
    ["numberbatch1908"]="numberbatch1908")
# Whether models are trained on lowercased tokens only (1) or not (0)
declare -A MODEL_LOWERCASED=(\
    ["glove"]=0 \
    ["numberbatch1908"]=1)
# Since the associative array above does not keep ordering, we need to specify
# it separately; this will also determine which models will be evaluated
EVALUATION_ORDER=( "glove" "numberbatch1908" )
# The original lexicon file; value of this variable will change after first filtering run
LEXICON="data/social-bias-lexicon.json"
EVALUATION_RESULTS_FILES=()
METRICS_OF_INTEREST=(\
    "ect" \
    "weat" \
    "rnsb")
WORD_LISTS_OF_INTEREST=(\
    "attribute" \
    "target")
TEST_TYPES_OF_INTEREST=(\
    "ethnicity" \
    "gender" \
    "religion")

SHUFFLED_RUNS=100
ATTRIBUTE_STEP_SIZE=6
TARGET_STEP_SIZE=2

PLOT_COLORS=("red" "blue")
OUTPUT_DIR="output/metric-evaluation/${TIMESTAMP}"

# Create output dir; it should not exist yet
echo "Creating ${OUTPUT_DIR} directory to save results to."
echo ""
mkdir -p "${OUTPUT_DIR}"


# ----------LEXICON FILTERING----------------------------------------

# Create filtered lexicon with all OOV tokens removed for all word embedding models
echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mFiltering lexicons for specified models.\e[0m"
for i in "${EVALUATION_ORDER[@]}"
do
    echo -e "\e[1;33m________________________________________\e[0m"
    echo -e "\e[1;33mFor '${i}' model...\e[0m"

    if [ "${MODEL_LOWERCASED[$i]}" == 0 ]
    then
        python filter_bias_lexicons.py \
            --embedding_model "${MODEL_PATHS[$i]}" \
            --lexicon_path "${LEXICON}" \
            --output "${OUTPUT_DIR}" \
            --output_filename "social-bias-lexicon--filtered--${TIMESTAMP}.json"
    else
        python filter_bias_lexicons.py \
            --embedding_model "${MODEL_PATHS[$i]}" \
            --lexicon_path "${LEXICON}" \
            --output "${OUTPUT_DIR}" \
            --output_filename "social-bias-lexicon--filtered--${TIMESTAMP}.json" \
            --lowercase
    fi

    # Re-set the lexicon path to the filtered one
    LEXICON="${OUTPUT_DIR}/social-bias-lexicon--filtered--${TIMESTAMP}.json"
done


# ----------METRIC EVALUATION----------------------------------------

echo ""
echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mEvaluating metrics with all specified models.\e[0m"
for i in "${EVALUATION_ORDER[@]}"
do

    echo -e "\e[1;33m________________________________________\e[0m"
    echo -e "\e[1;33mFor '${i}' model...\e[0m"

    if [ "${MODEL_LOWERCASED[$i]}" == 0 ]
    then
        python metric_evaluation.py \
            --embedding_model "${MODEL_PATHS[$i]}" \
            --lexicon_path "${LEXICON}" \
            --output "${OUTPUT_DIR}" \
            --output_filename "metric-evaluation-${TIMESTAMP}-${i}.json" \
            --shuffled_runs ${SHUFFLED_RUNS} \
            --attribute_step_size ${ATTRIBUTE_STEP_SIZE} \
            --target_step_size ${TARGET_STEP_SIZE} \
            --allow_different_lexicon_sizes
    else
        python metric_evaluation.py \
            --embedding_model "${MODEL_PATHS[$i]}" \
            --lexicon_path "${LEXICON}" \
            --output "${OUTPUT_DIR}" \
            --output_filename "metric-evaluation-${TIMESTAMP}-${i}.json" \
            --shuffled_runs ${SHUFFLED_RUNS} \
            --attribute_step_size ${ATTRIBUTE_STEP_SIZE} \
            --target_step_size ${TARGET_STEP_SIZE} \
            --allow_different_lexicon_sizes \
            --lowercase
    fi
    EVALUATION_RESULTS_FILES=(\
        "${EVALUATION_RESULTS_FILES[@]}" \
        "${OUTPUT_DIR}/metric-evaluation-${TIMESTAMP}-${i}.json")
done


# ----------EVALUATION RESULTS ANALYSIS----------------------------------------

echo ""
echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mAnalyzing evaluation results.\e[0m"

python analyze_metric_evaluation_results.py \
    --evaluation_results "${EVALUATION_RESULTS_FILES[@]}" \
    --output "${OUTPUT_DIR}" \
    --output_filename "evaluation-analysis-${TIMESTAMP}.json"


# ----------RESULTS PLOTTING----------------------------------------

echo ""
echo -e "\e[1;31m============================================\e[0m"
echo -e "\e[1;31mPlotting results.\e[0m"

for i in "${TEST_TYPES_OF_INTEREST[@]}"
do
    for j in "${WORD_LISTS_OF_INTEREST[@]}"
    do

        echo -e "\e[1;33m________________________________________\e[0m"
        echo -e "\e[1;33mFor test type '${i}' and '${j}' word lists...\e[0m"

        python plot_metric_evaluation.py \
            --results_file "${OUTPUT_DIR}/evaluation-analysis-${TIMESTAMP}.json" \
            --metrics "${METRICS_OF_INTEREST[@]}" \
            --output "${OUTPUT_DIR}" \
            --plot_colors "${PLOT_COLORS[@]}" \
            --shuffle_type "$j" \
            --test_type "$i" \
            --plot_individual_runs
    done
done

echo -e "\e[1;31mMetric analysis pipeline finished.\e[0m"
