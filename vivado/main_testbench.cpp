#include <stdlib.h>

#include <iostream>

#include "fileio.hpp"
#include "defs.h"
#include "utils.hpp"

// #include "sgd_baseline.h"

#include "sgd_serial_kernel.h"

int main(int argc, char *argv[]) {

    // Output file that saves the test bench results
    std::ofstream outfile;
    outfile.open("out.dat");

    // Initialize options to default values
    std::string path_to_data("/work/zhang/users/gaa54/Spam-Filter/data");

    // Verify that the path to data exists
    if(access(path_to_data.c_str(), F_OK) == -1) {
        std::cout << "Couldn't find data directory" << std::endl;
        return EXIT_FAILURE;
    }

    // Read data from files and insert into variables
    std::string str_points_filepath(
        path_to_data + std::string("/shuffledfeats.dat"));
    std::string str_labels_filepath(
        path_to_data + std::string("/shuffledlabels.dat"));

    const char* points_filepath = str_points_filepath.c_str();
    const char* labels_filepath = str_labels_filepath.c_str();

    // Model variables
    FeatureType data_points[DATA_SET_SIZE]; // Data points
    LabelType labels[NUM_SAMPLES]; // Labels
    FeatureType parameter_vector[NUM_FEATURES]; // Model parameters

    // Read data from files and insert into variables
    if(readData(NUM_SAMPLES * NUM_FEATURES, points_filepath, data_points)
        != (NUM_SAMPLES * NUM_FEATURES))
            return EXIT_FAILURE;

    if(readData(NUM_SAMPLES, labels_filepath, labels)
        != NUM_SAMPLES)
            return EXIT_FAILURE;

    // memset(parameter_vector, 0, NUM_FEATURES * sizeof(FeatureType));
    for (size_t i = 0; i < NUM_FEATURES; i++)
        parameter_vector[i] = 0.5;

    SgdLR(data_points, labels, parameter_vector);

    std::cout << "\nmain parameter vector: \n";
    for(int i=0;i<30;i++)
        std::cout << "m" << i << ":" << parameter_vector[i] << " | " ;
    std::cout << std::endl;

    // Initialize benchmark variables
    double training_tpr = 0.0;
    double training_fpr = 0.0;
    double training_error = 0.0;
    double testing_tpr = 0.0;
    double testing_fpr = 0.0;
    double testing_error = 0.0;

    // Get Training error
    DataSet training_set;
    training_set.data_points = data_points;
    training_set.labels = labels;
    training_set.num_data_points = NUM_TRAINING;
    training_set.num_features = NUM_FEATURES;
    training_set.parameter_vector = parameter_vector;
    computeErrorRate(training_set, &training_tpr, &training_fpr, &training_error);

    // Get Testing error
    DataSet testing_set;
    testing_set.data_points = &data_points[NUM_FEATURES * NUM_TRAINING];
    testing_set.labels = &labels[NUM_TRAINING];
    testing_set.num_data_points = NUM_TESTING;
    testing_set.num_features = NUM_FEATURES;
    testing_set.parameter_vector = parameter_vector;
    computeErrorRate(testing_set, &testing_tpr, &testing_fpr, &testing_error);

    training_tpr *= 100.0;
    training_fpr *= 100.0;
    training_error *= 100.0;
    testing_tpr *= 100.0;
    testing_fpr *= 100.0;
    testing_error *= 100.0;

    outfile << "train TPR,train FPR,train Error,test TPR,test FPR,test Error\n";
    outfile
        << training_tpr << ","
        << training_fpr << ","
        << training_error << ","
        << testing_tpr << ","
        << testing_fpr << ","
        << testing_error << std::endl;

    std::cout << "train TPR,train FPR,train Error,test TPR,test FPR,test Error\n";
    std::cout
        << training_tpr << ","
        << training_fpr << ","
        << training_error << ","
        << testing_tpr << ","
        << testing_fpr << ","
        << testing_error << std::endl;

    // Free memory and exit
    // delete[] data_points;
    // delete[] labels;
    // delete[] parameter_vector;

    // Close output file
    outfile.close();

    return 0;

}
