#include <stdlib.h>

#include "utils/mnist_fileio.hpp"
#include "utils/mnist_defs.h"
#include "utils/mnist_utils.hpp"
#include "sgd_baseline.h"
#include "batch_baseline.h"
#include "minibatch_baseline.h"
// Run the code for a fixed number of epochs 
//and output time and error rates
void runTrainAndTest(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    trainAndTest(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    trainAndTest(
        trainBatchGradientDescent,
        "BGD",
        data_set,
        training_options,
        benchmark_options);
    
    size_t batch_sizes[10] = {1,2,5,10,20,50,100,200,500,1000};
    for (size_t i = 0; i < 10; i++) {
        training_options.config_params["batch_size"]
        = batch_sizes[i];
        trainAndTest(
            trainMiniBatchGradientDescent,
            "MBGD",
            data_set,
            training_options,
            benchmark_options);
    }

}
// runs the training n epochs at a time
//until it reaches m epochs, printing error
//rate every n epochs 
void runConvergenceRate(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    convergenceRate(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    convergenceRate(
        trainBatchGradientDescent,
        "BGD",
        data_set,
        training_options,
        benchmark_options);

    size_t batch_sizes[10] = {1,2,5,10,20,50,100,200,500,1000};
    
    for (size_t i = 0; i < 10; i++) {
        training_options.config_params["batch_size"]
        = batch_sizes[i];
        convergenceRate(
            trainMiniBatchGradientDescent,
            "MBGD",
            data_set,
            training_options,
            benchmark_options);
    }

}
//run the training one epoch at a time until some accurancy
//goal is reached or reach maximum epochs
void runConvergenceTime(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    // convergenceTime(
    //     trainStochasticGradientDescent,
    //     "SGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);

    // convergenceTime(
    //     trainBatchGradientDescent,
    //     "BGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);

    size_t batch_sizes[10] = {1,2,5,10,20,50,100,200,500,1000};
    for (size_t i = 0; i < 10; i++) {
        training_options.config_params["batch_size"]
        = batch_sizes[i];
        convergenceTime(
            trainMiniBatchGradientDescent,
            "MBGD",
            data_set,
            training_options,
            benchmark_options);
    }

}


int main(int argc, char *argv[]) {

    // Initialize options to default values
    double* step_size = new double;
    TrainingOptions training_options = initDefaultTrainingOptions(step_size);
    BenchmarkOptions benchmark_options = initDefaultBenchmarkOptions();
    std::string path_to_data("data");

    // Parse arguments to adjust options
    parse_command_line_args(argc, argv, &training_options, &benchmark_options, &path_to_data);

    printf("in main error goal is %f\n",benchmark_options.error_goal);
    printf("in main run number is %d\n",benchmark_options.num_runs);
    printf("in main step size is %f \n",*training_options.step_size);

    // Model variables
    FeatureType* data_points = new FeatureType[DATA_SET_SIZE]; // Data points
    LabelType* labels = new LabelType[NUM_SAMPLES]; // Labels
    FeatureType* parameter_vector = new FeatureType[PARAMETER_SIZE]; // Model parameters


    //build train file path
    std::string str_train_points_filepath(path_to_data + 
    						std::string("/train-images-idx3-ubyte"));
    const char* train_points_filepath = str_train_points_filepath.c_str();
    std::string str_train_labels_filepath(path_to_data + 
    						std::string("/train-labels-idx1-ubyte"));
    const char* train_labels_filepath = str_train_labels_filepath.c_str();
    //build test file path
    std::string str_test_points_filepath(path_to_data + 
    						std::string("/t10k-images-idx3-ubyte"));
    const char* test_points_filepath = str_test_points_filepath.c_str();
    std::string str_test_labels_filepath(path_to_data + 
    						std::string("/t10k-labels-idx1-ubyte"));
    const char* test_labels_filepath = str_test_labels_filepath.c_str();


    // Read train data from files and insert into variables
    if( readImageData (TRAIN_SET_SIZE, train_points_filepath, 
    								data_points) != TRAIN_SET_SIZE) 
                                        return EXIT_FAILURE;
    if( readImageLables(NUM_TRAINING, train_labels_filepath, 
    								labels) != NUM_TRAINING)   
                                        return EXIT_FAILURE;
    //Read test data from files and insert behind
    if(readImageData(TEST_SET_SIZE, test_points_filepath,
				&data_points[TRAIN_SET_SIZE])!= TEST_SET_SIZE)  
                                        return EXIT_FAILURE;
    if(readImageLables(NUM_TESTING, test_labels_filepath,
						 &labels[NUM_TRAINING])!= NUM_TESTING)    
                                        return EXIT_FAILURE;
    
    cout<<"read mnist data success"<<endl;

    // Initialize data set and options structs
    DataSet data_set;
    data_set.data_points = data_points;
    data_set.labels = labels;
    data_set.parameter_vector = parameter_vector;
    data_set.num_data_points = NUM_SAMPLES;
    data_set.num_features = NUM_FEATURES;

    // Initial shuffle of the data set to mix spam with ham
    // shuffleKeyValue(data_points, labels, NUM_SAMPLES, NUM_FEATURES);



   //  runConvergenceRate(data_set, training_options, benchmark_options);
   // runTrainAndTest(data_set, training_options, benchmark_options);
      runConvergenceTime(data_set, training_options, benchmark_options);


    // Free memory and exit
    delete   step_size;
    delete[] data_points;
    delete[] labels;
    delete[] parameter_vector;

    return 0;
}
