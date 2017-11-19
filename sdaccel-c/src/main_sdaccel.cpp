#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string.h>
#include <unistd.h>

#include "fileio.hpp"
#include "defs.h"
#include "utils.hpp"
#include "SdaccelSgd.h"

static void print_usage(void)
{
    std::cout << "usage: %s <options>\n";
    std::cout << "  -d <cpu|gpu|acc>\n";
    std::cout << "  -k [kernel name]\n";
    std::cout << "  -f [kernel file]\n";
}

static int parse_sdaccel_command_line_args(
    int argc,
    char** argv,
    cl_device_type* deviceType,
    std::string* kernelName,
    std::string* kernelFile) {

    int c = 0;

    while ((c = getopt(argc, argv, "d:k:f:")) != -1) {
        switch (c) {
        case 'd':
            if (strcmp(optarg, "gpu") == 0)
                *deviceType = CL_DEVICE_TYPE_GPU;
            else if (strcmp(optarg, "cpu") == 0)
                *deviceType = CL_DEVICE_TYPE_CPU;
            else if (strcmp(optarg, "acc") == 0)
                *deviceType = CL_DEVICE_TYPE_ACCELERATOR;
            else {
                print_usage();
                return -1;
            }
            break;
        case 'k':
            *kernelName = optarg;
            break;
        case 'f':
            *kernelFile = optarg;
            break;
        default:
            print_usage();
        } // matching on arguments
    } // while args present
}

int main(int argc, char *argv[]) {
    printf("Entered main\n");

    // Initialize options to default values
    cl_device_type deviceType = CL_DEVICE_TYPE_ACCELERATOR;
    std::string kernelName("");
    std::string kernelFile("");
    std::string path_to_data("/work/zhang/users/gaa54/Spam-Filter/data");

    // Parse arguments to adjust options
    parse_sdaccel_command_line_args(argc, argv, &deviceType, &kernelName,
                                    &kernelFile);

    // Verify that the path to data exists
    if(access(path_to_data.c_str(), F_OK) == -1) {
        std::cout << "Couldn't find data directory" << std::endl;
        return EXIT_FAILURE;
    }

    /****************ALLOCATION AND INITIALIZATION OF DATASET******************/

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

    /**************************HARNESS STUFF***********************************/
    SdaccelSgd spamfilter_kernel (kernelFile.c_str(), kernelName.c_str(),
        deviceType, data_points, labels, parameter_vector);
    fflush(stdout);

    spamfilter_kernel.load_to_memory();
    fflush(stdout);

    spamfilter_kernel.run_kernel();
    fflush(stdout);

    spamfilter_kernel.check_results();
    spamfilter_kernel.finish_and_clean_up();

    /************************CLEANUP OF MAIN VARIABLES*************************/
    // free(data_points);
    // free(labels);
    // free(parameter_vector);

    return 0;
}
