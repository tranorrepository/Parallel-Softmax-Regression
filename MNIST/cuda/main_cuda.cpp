#include <stdlib.h>

#include "utils/mnist_fileio.hpp"
#include "utils/mnist_defs.h"
#include "utils/mnist_utils.hpp"
#include "cpp/sgd_baseline.h"
#include "cpp/batch_baseline.h"
#include "cpp/minibatch_baseline.h"
#include "mbgd_1.h"
#include "mbgd_2.h"
// #include "sgd_cublas.h"
#include "sgd_single_point_1.h"


void runTrainAndTest(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    //baseline SGD
    trainAndTest(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    // //baseline BGD
    trainAndTest(
        trainBatchGradientDescent,
        "BGD",
        data_set,
        training_options,
        benchmark_options);

    // //baseline MBGD
    size_t batch_sizes1[10] = {1,2,5,10,20,50,100,200,500,1000};
    for (size_t i = 0; i < 10; i++) {
        training_options.config_params["batch_size"]
        = batch_sizes1[i];
        trainAndTest(
            trainMiniBatchGradientDescent,
            "MBGD",
            data_set,
            training_options,
            benchmark_options);
    }

    //cublas+cuda implementation
    // much slower than baseline code
    // trainAndTest(
    //     trainStochasticGradientDescent3,
    //     "Cublas SGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);


    // blocking parameter
    // Be factor of LABLE_CLASS
    size_t threads_class_per_datapoint[4] = {1, 2, 5, 10};

    //CUDA sgd implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){
        //define threads classes for one datapoint for blocking
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_datapoint = 32;
            threads_per_datapoint <= 512;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t datapoints_per_block = 1;
                datapoints_per_block <= 8;
                datapoints_per_block *= 2) {

                training_options.config_params["datapoints_per_block"]
                = datapoints_per_block;

                trainAndTest(
                    trainParallelStochasticGradientDescent1,
                    "CUDA SGD",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    for(size_t i = 2 ;i < 4; i++){
        //define threads classes for one datapoint for blocking
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_datapoint = 40;
            threads_per_datapoint <= 640;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t datapoints_per_block = 1;
                datapoints_per_block <= 8;
                datapoints_per_block *= 2) {

                training_options.config_params["datapoints_per_block"]
                = datapoints_per_block;

                trainAndTest(
                    trainParallelStochasticGradientDescent1,
                    "CUDA SGD",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    // CUDA MBGD-1 implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){

        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        size_t batch_sizes2[8] = {2, 4, 6, 8, 8, 9, 10, 11};
        for (size_t threads_per_datapoint = 32;
            threads_per_datapoint <= 512;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes2[i];

                trainAndTest(
                    trainParallelMiniBatchGradientDescent,
                    "CUDA MBGD 1",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    for(size_t i = 2 ;i < 4; i++){

        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        size_t batch_sizes2[8] = {2, 3, 4, 6, 8, 8, 9, 10};
        for (size_t threads_per_datapoint = 40;
            threads_per_datapoint <= 640;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes2[i];

                trainAndTest(
                    trainParallelMiniBatchGradientDescent,
                    "CUDA MBGD 1",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    
    // CUDA MBGD-2 implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){
        size_t batch_sizes4[5] = {2, 4, 6, 8, 40};
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_mini_batch = 32;
                 threads_per_mini_batch <= 512;
                 threads_per_mini_batch *= 2) {
        
            training_options.config_params["threads_per_mini_batch"]
            =  threads_per_mini_batch;
        
            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes4[i];
        
                trainAndTest(
                    trainParallelMiniBatchGradientDescent2,
                    "CUDA MBGD 2",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }
   
    for(size_t i = 2 ;i < 4; i++){
        size_t batch_sizes4[5] = {2, 3, 4, 6, 8};
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_mini_batch = 40;
                 threads_per_mini_batch <= 640;
                 threads_per_mini_batch *= 2) {
        
            training_options.config_params["threads_per_mini_batch"]
            =  threads_per_mini_batch;
        
            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes4[i];
        
                trainAndTest(
                    trainParallelMiniBatchGradientDescent2,
                    "CUDA MBGD 2",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }


}

void runConvergenceRate(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    //baseline SGD
    convergenceRate(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    //baseline BGD
    convergenceRate(
        trainBatchGradientDescent,
        "BGD",
        data_set,
        training_options,
        benchmark_options);

    //baseline MBGD
    size_t batch_sizes1[10] = {1,2,5,10,20,50,100,200,500,1000};
    for (size_t i = 0; i < 10; i++) {
        training_options.config_params["batch_size"]
        = batch_sizes1[i];
        convergenceRate(
            trainMiniBatchGradientDescent,
            "MBGD",
            data_set,
            training_options,
            benchmark_options);
    }

    //cublas+cuda implementation
    // much slower than baseline code
    // convergenceRate(
    //     trainStochasticGradientDescent3,
    //     "Cublas SGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);


    // blocking parameter
    // Be factor of LABLE_CLASS
    size_t threads_class_per_datapoint[4] = {1, 2, 5, 10};

    //CUDA sgd implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){
        //define threads classes for one datapoint for blocking
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_datapoint = 32;
            threads_per_datapoint <= 512;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t datapoints_per_block = 1;
                datapoints_per_block <= 8;
                datapoints_per_block *= 2) {

                training_options.config_params["datapoints_per_block"]
                = datapoints_per_block;

                convergenceRate(
                    trainParallelStochasticGradientDescent1,
                    "CUDA SGD",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    for(size_t i = 2 ;i < 4; i++){
        //define threads classes for one datapoint for blocking
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_datapoint = 40;
            threads_per_datapoint <= 640;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t datapoints_per_block = 1;
                datapoints_per_block <= 8;
                datapoints_per_block *= 2) {

                training_options.config_params["datapoints_per_block"]
                = datapoints_per_block;

                convergenceRate(
                    trainParallelStochasticGradientDescent1,
                    "CUDA SGD",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    // CUDA MBGD-1 implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){

        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        size_t batch_sizes2[8] = {2, 4, 6, 8, 8, 9, 10, 11};
        for (size_t threads_per_datapoint = 32;
            threads_per_datapoint <= 512;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes2[i];

                convergenceRate(
                    trainParallelMiniBatchGradientDescent,
                    "CUDA MBGD 1",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    for(size_t i = 2 ;i < 4; i++){

        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        size_t batch_sizes2[8] = {2, 3, 4, 6, 8, 8, 9, 10};
        for (size_t threads_per_datapoint = 40;
            threads_per_datapoint <= 640;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes2[i];

                convergenceRate(
                    trainParallelMiniBatchGradientDescent,
                    "CUDA MBGD 1",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    
    // CUDA MBGD-2 implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){
        size_t batch_sizes4[5] = {2, 4, 6, 8, 40};
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_mini_batch = 32;
                 threads_per_mini_batch <= 512;
                 threads_per_mini_batch *= 2) {
        
            training_options.config_params["threads_per_mini_batch"]
            =  threads_per_mini_batch;
        
            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes4[i];
        
                convergenceRate(
                    trainParallelMiniBatchGradientDescent2,
                    "CUDA MBGD 2",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }
   
    for(size_t i = 2 ;i < 4; i++){
        size_t batch_sizes4[5] = {2, 3, 4, 6, 8};
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_mini_batch = 40;
                 threads_per_mini_batch <= 640;
                 threads_per_mini_batch *= 2) {
        
            training_options.config_params["threads_per_mini_batch"]
            =  threads_per_mini_batch;
        
            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes4[i];
        
                convergenceRate(
                    trainParallelMiniBatchGradientDescent2,
                    "CUDA MBGD 2",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

}

void runConvergenceTime(
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options) {

    //baseline SGD
    convergenceTime(
        trainStochasticGradientDescent,
        "SGD",
        data_set,
        training_options,
        benchmark_options);

    // //baseline BGD
    // convergenceTime(
    //     trainBatchGradientDescent,
    //     "BGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);

    // //baseline MBGD
    // size_t batch_sizes1[10] = {1,2,5,10,20,50,100,200,500,1000};
    // for (size_t i = 0; i < 10; i++) {
    //     training_options.config_params["batch_size"]
    //     = batch_sizes1[i];
    //     convergenceTime(
    //         trainMiniBatchGradientDescent,
    //         "MBGD",
    //         data_set,
    //         training_options,
    //         benchmark_options);
    // }

    //cublas+cuda implementation
    // much slower than baseline code
    // convergenceTime(
    //     trainStochasticGradientDescent3,
    //     "Cublas SGD",
    //     data_set,
    //     training_options,
    //     benchmark_options);


    // blocking parameter
    // Be factor of LABLE_CLASS
    size_t threads_class_per_datapoint[4] = {1, 2, 5, 10};

    // //CUDA sgd implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){
        //define threads classes for one datapoint for blocking
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_datapoint = 32;
            threads_per_datapoint <= 512;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t datapoints_per_block = 1;
                datapoints_per_block <= 8;
                datapoints_per_block *= 2) {

                training_options.config_params["datapoints_per_block"]
                = datapoints_per_block;

                convergenceTime(
                    trainParallelStochasticGradientDescent1,
                    "CUDA SGD",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    for(size_t i = 2 ;i < 4; i++){
        //define threads classes for one datapoint for blocking
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_datapoint = 40;
            threads_per_datapoint <= 640;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t datapoints_per_block = 1;
                datapoints_per_block <= 8;
                datapoints_per_block *= 2) {

                training_options.config_params["datapoints_per_block"]
                = datapoints_per_block;

                convergenceTime(
                    trainParallelStochasticGradientDescent1,
                    "CUDA SGD",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }


    // CUDA MBGD-1 implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){

        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        size_t batch_sizes2[8] = {2, 4, 6, 8, 8, 9, 10, 11};
        for (size_t threads_per_datapoint = 32;
            threads_per_datapoint <= 512;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes2[i];

                convergenceTime(
                    trainParallelMiniBatchGradientDescent,
                    "CUDA MBGD 1",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    for(size_t i = 2 ;i < 4; i++){

        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        size_t batch_sizes2[8] = {2, 3, 4, 6, 8, 8, 9, 10};
        for (size_t threads_per_datapoint = 40;
            threads_per_datapoint <= 640;
            threads_per_datapoint *= 2) {

            training_options.config_params["threads_per_datapoint"]
            = threads_per_datapoint;

            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes2[i];

                convergenceTime(
                    trainParallelMiniBatchGradientDescent,
                    "CUDA MBGD 1",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

    
    // CUDA MBGD-2 implementation with shared memory
    for(size_t i = 0 ;i < 2; i++){
        size_t batch_sizes4[5] = {2, 4, 6, 8, 40};
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_mini_batch = 32;
                 threads_per_mini_batch <= 512;
                 threads_per_mini_batch *= 2) {
        
            training_options.config_params["threads_per_mini_batch"]
            =  threads_per_mini_batch;
        
            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes4[i];
        
                convergenceTime(
                    trainParallelMiniBatchGradientDescent2,
                    "CUDA MBGD 2",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }
   
    for(size_t i = 2 ;i < 4; i++){
        size_t batch_sizes4[5] = {2, 3, 4, 6, 8};
        training_options.config_params["threads_class_per_datapoint"]
                = threads_class_per_datapoint[i];

        for (size_t threads_per_mini_batch = 40;
                 threads_per_mini_batch <= 640;
                 threads_per_mini_batch *= 2) {
        
            training_options.config_params["threads_per_mini_batch"]
            =  threads_per_mini_batch;
        
            for (size_t i = 0;
                i < 4;
                i++) {
                
                training_options.config_params["batch_size"]
                = batch_sizes4[i];
        
                convergenceTime(
                    trainParallelMiniBatchGradientDescent2,
                    "CUDA MBGD 2",
                    data_set,
                    training_options,
                    benchmark_options);
            }
        }
    }

}

int main(int argc, char *argv[]) {

    // Initialize options to default values
    double* step_size = new double;
    TrainingOptions training_options = initDefaultTrainingOptions(step_size);
    BenchmarkOptions benchmark_options = initDefaultBenchmarkOptions();
    std::string path_to_data("data");

    // Parse arguments to adjust options
    parse_command_line_args(argc,
        argv,
        &training_options,
        &benchmark_options,
        &path_to_data);

    // Model variables
    FeatureType* data_points = new FeatureType[DATA_SET_SIZE];
    LabelType* labels = new LabelType[NUM_SAMPLES];
    FeatureType* parameter_vector = new FeatureType[PARAMETER_SIZE];

    ///build train file path
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
    if(readImageData (TRAIN_SET_SIZE, train_points_filepath, 
                                    data_points) != TRAIN_SET_SIZE) 
                                        return EXIT_FAILURE;
    if(readImageLables(NUM_TRAINING, train_labels_filepath, 
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

    // runConvergenceRate(data_set, training_options, benchmark_options);
    // runTrainAndTest(data_set, training_options, benchmark_options);
    runConvergenceTime(data_set, training_options, benchmark_options);

    // Free memory and exit
    delete[] data_points;
    delete[] labels;
    delete[] parameter_vector;
    delete step_size;

    return 0;
}
