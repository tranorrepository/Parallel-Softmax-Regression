#include "SdaccelSgd.h"
#include "utils.hpp"
#include <string.h>
#include <iostream>

size_t SdaccelSgd::num_points_training = NUM_TRAINING;
size_t SdaccelSgd::num_points_testing = NUM_TESTING;
size_t SdaccelSgd::num_features = NUM_FEATURES;

SdaccelSgd::SdaccelSgd(
    const char *filename,
    const char* kernel_name,
    cl_device_type device_type,
    FeatureType* data_points,
    LabelType* labels,
    FeatureType* parameter_vector)
    :
    CLKernel(filename, kernel_name, device_type),
    data_points(data_points),
    labels(labels),
    parameter_vector(parameter_vector) {
    // empty child constructor
}

int SdaccelSgd::setup_kernel() {
    set_global(GLOBAL_WORK_GROUP_SIZE); //sets global work group size
    set_local(LOCAL_WORK_GROUP_SIZE);  //sets local work group size
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    return EXIT_SUCCESS;
}

int SdaccelSgd::load_data() {
    size_t num_data_points = num_points_training + num_points_testing;

    // Create and set up the memory objects using the new CLMemObj class
    // TODO: need to find a proper place to put this; need static because check_results also needs
    // mem buffers; if we don't have static, the variable gets deleted when the function ends
    // maybe we need to have the CLKernel harness to take charge of the memory objects management
    memobj_array.push_back(CLMemObj(data_points     , sizeof(FeatureType), num_data_points * num_features, CL_MEM_READ_ONLY ));
    memobj_array.push_back(CLMemObj(labels          , sizeof(LabelType)  , num_data_points               , CL_MEM_READ_ONLY ));
    memobj_array.push_back(CLMemObj(parameter_vector, sizeof(FeatureType), num_features                  , CL_MEM_READ_WRITE));

    return EXIT_SUCCESS;
}

bool SdaccelSgd::check_results() {
    const unsigned int idx_of_parameter_vector_in_memobj_array = 2;

    // Read back the results from the device to verify the output
    read_clmem(idx_of_parameter_vector_in_memobj_array, sizeof(FeatureType), num_features, parameter_vector);

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

    // outfile << "train TPR,train FPR,train Error,test TPR,test FPR,test Error\n";
    // outfile
    //     << training_tpr << ","
    //     << training_fpr << ","
    //     << training_error << ","
    //     << testing_tpr << ","
    //     << testing_fpr << ","
    //     << testing_error << std::endl;

    std::cout << "train TPR,train FPR,train Error,test TPR,test FPR,test Error\n";
    std::cout
        << training_tpr << ","
        << training_fpr << ","
        << training_error << ","
        << testing_tpr << ","
        << testing_fpr << ","
        << testing_error << std::endl;

    return /*correct == RESULTS_SIZE ?*/ true /*: false*/;
}

void SdaccelSgd::clean_up() {
    // Nothing to clean up for now
}
