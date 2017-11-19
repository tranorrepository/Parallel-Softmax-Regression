#include <math.h>
#include <getopt.h>
#include <string.h>

#include "utils.hpp"



/******************************************************************************/
/*                                                                            */
/* General Utilities                                                          */
/*                                                                            */
/******************************************************************************/

// initialize default configuration options
TrainingOptions initDefaultTrainingOptions() {

    Dictionary default_config_params;
    default_config_params["step_size"] = STEP_SIZE;

    TrainingOptions default_training_options = {NUM_EPOCHS, default_config_params};

    return default_training_options;
}
BenchmarkOptions initDefaultBenchmarkOptions() {

    static const BenchmarkOptions default_benchmark_options = {NUM_TRAINING, NUM_TESTING, MAX_NUM_EPOCHS, NUM_RUNS, ERROR_GOAL};

    return default_benchmark_options;
}


// prints usage in case an invalid command line argument is passed
static void print_usage()
{
    printf("usage: <name_of_binary> <options>\n");
    printf("options:\n");
    printf("    -e num_epochs\n");
    printf("    -z step_size\n");
    printf("    -t num_points_training\n");
    printf("    -s num_points_testing\n");
    printf("    -m max_num_epochs\n");
    printf("    -r num_runs\n");
    printf("    -g error_goal\n");
    printf("    -p path_to_data_directory\n");
}

// parses training- and benchmark- related command line arguments
void parse_command_line_args(int argc, char ** argv, TrainingOptions* training_options, BenchmarkOptions* benchmark_options, std::string* path_to_data) {
    int c = 0;

    while ((c = getopt(argc, argv, "e:z:t:s:m:r:g:p:")) != -1) {
        switch (c) {
            case 'e':
                if (training_options) (*training_options).num_epochs = atol(optarg);
                break;
            case 'z':
                if (training_options) (*training_options).config_params["step_size"] = atof(optarg);
                break;
            case 't':
                if (benchmark_options) (*benchmark_options).num_points_training = atol(optarg);
                break;
            case 's':
                if (benchmark_options) (*benchmark_options).num_points_testing = atol(optarg);
                break;
            case 'm':
                if (benchmark_options) (*benchmark_options).max_num_epochs = atol(optarg);
                break;
            case 'r':
                if (benchmark_options) (*benchmark_options).num_runs = atoi(optarg);
                break;
            case 'g':
                if (benchmark_options) (*benchmark_options).error_goal = atof(optarg);
                break;
            case 'p':
                if (path_to_data) *path_to_data = optarg;
                break;
            default:
                print_usage();
        } // matching on arguments
    } // while args present
}




/******************************************************************************/
/*                                                                            */
/* Logistic Regression-Specific Utilities                                     */
/*                                                                            */
/******************************************************************************/

// computes logistic function for a given double
double logisticFunction(double exponent) {
    return (1.0 / (1.0 + exp(-exponent)));
}

// computes logistic function for a given parameter vector (parameter_vector) and a data point (data_point_i)
double logisticFunction(FeatureType* parameter_vector, FeatureType* data_point_i, const size_t num_features) {
    return logisticFunction(dotProduct(parameter_vector, data_point_i, num_features));
}


// gets predicted label for a given parameter vector (parameter_vector) and a given datapoint (data_point_i)
LabelType getPrediction(FeatureType* parameter_vector, FeatureType* data_point_i, size_t num_features, const double treshold = 0) {
    float parameter_vector_dot_x_i = dotProduct(parameter_vector, data_point_i, num_features);
    return (parameter_vector_dot_x_i > treshold) ? POSITIVE_LABEL : NEGATIVE_LABEL;
}


// serial implementation of accuracy test
// computes true positive rate, false positive rate and error rate, then adds
// these values to variables passed through pointers
double computeErrorRate(
    DataSet data_set,
    double* cumulative_true_positive_rate,
    double* cumulative_false_positive_rate,
    double* cumulative_error){

    size_t true_positives = 0, true_negatives = 0, false_positives = 0, false_negatives = 0;
    for (size_t i = 0; i < data_set.num_data_points; i++) {
        LabelType prediction = getPrediction(data_set.parameter_vector, &data_set.data_points[i * data_set.num_features], data_set.num_features);
        if (prediction != data_set.labels[i]){
            if (prediction == POSITIVE_LABEL)
                false_positives++;
            else
                false_negatives++;
        } else {
            if (prediction == POSITIVE_LABEL)
                true_positives++;
            else
                true_negatives++;
        }
    }
    double error_rate = (double)(false_positives + false_negatives) / data_set.num_data_points;
    if (cumulative_true_positive_rate != NULL) *cumulative_true_positive_rate += (double)true_positives / (true_positives + false_negatives);
    if (cumulative_false_positive_rate != NULL) *cumulative_false_positive_rate += (double)false_positives / (false_positives + true_negatives);
    if (cumulative_error != NULL) *cumulative_error += error_rate;

    return error_rate;
}

// loss function for the whole batch (Negative LogLikelihood for Bernoulli probability distribution)
double lossFunction(DataSet data_set) {
    double sum = 0.0;
    for (size_t i = 0; i < data_set.num_data_points; i++) {
        double probability_of_positive = logisticFunction(data_set.parameter_vector, &data_set.data_points[i * data_set.num_features], data_set.num_features);
        FeatureType label_i = (FeatureType) data_set.labels[i];
        sum += ((label_i * log(probability_of_positive)) + ((1 - label_i) * log(1 - probability_of_positive))) / data_set.num_data_points;
    }
    return -sum; // negative of the sum
}

// Makes a printable representation of the list of configuration parameters
static std::string configParamsToStr(Dictionary config_params) {
    std::string result("");
    for (Dictionary::iterator param = config_params.begin(); param != config_params.end(); ++param) {
        char second[100];
        sprintf(second, "%g", param->second);
        result += param->first + ":" + second + "&";
    }
    return result.substr(0, result.size()-1);
}
