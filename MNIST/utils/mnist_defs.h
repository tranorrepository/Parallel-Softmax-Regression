#ifndef MNIST_DEFS
#define MNIST_DEFS

#include <string>
#include <map>

// constants and default values for some configuration parameters
#define NUM_FEATURES      784
#define NUM_SAMPLES       70000
#define DATA_SET_SIZE     (NUM_FEATURES * NUM_SAMPLES)
#define NUM_TRAINING      60000
#define NUM_TESTING       (NUM_SAMPLES - NUM_TRAINING)
#define TRAIN_SET_SIZE    (NUM_FEATURES * NUM_TRAINING) 
#define TEST_SET_SIZE     (NUM_FEATURES * NUM_TESTING) 
#define NUM_VALIDATION    0
#define LAMBDA            0 //regularization parameter
#define STEP_SIZE         0.013 //step size (eta) 
#define EPSILON           0.0000001
#define MAX_ITER          (NUM_TRAINING)
#define TOLERANCE         0.0001
#define NUM_EPOCHS        30
#define MAX_NUM_EPOCHS    100
#define NUM_RUNS          2
#define ERROR_GOAL        0.07
#define THREADS_PER_DATAPOINT    256
#define DATAPOINTS_PER_BLOCK     2
#define THREADS_CLASS_PER_DATAPOINT 2
#define CHARACTERISTIC_TIME (MAX_NUM_EPOCHS * NUM_TRAINING / 3)
#define BATCH_SIZE        60
#define THREADS_PER_MINI_BATCH 256 
#define POSITIVE_LABEL    1
#define NEGATIVE_LABEL    0
#define LABEL_CLASS       10
#define PARAMETER_SIZE    (LABEL_CLASS * NUM_FEATURES)

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif


// type definitions

#ifdef USE_OPENBLAS
    // must be float when using BLAS
    typedef float FeatureType;
    typedef float LabelType;
#else
    // might be some other type if not using BLAS
    typedef float FeatureType;
    typedef float LabelType;
#endif
typedef std::map<std::string, double> Dictionary;

struct DataSet_s {
    FeatureType* data_points;
    LabelType* labels;
    FeatureType* parameter_vector;
    size_t num_data_points;
    size_t num_features;
};
typedef struct DataSet_s DataSet;
struct TrainingOptions_s {
    size_t num_epochs;
    double* step_size;
    Dictionary config_params;
};
typedef struct TrainingOptions_s TrainingOptions;
struct BenchmarkOptions_s {
    size_t num_points_training;
    size_t num_points_testing;
    size_t max_num_epochs;
    unsigned int num_runs;
    double error_goal;
};
typedef struct BenchmarkOptions_s BenchmarkOptions;

// other configuration parameters
#define PRINT_AS_CSV      true
// #define USE_OPENBLAS //comment out this line to disable openblas

// colors for text output
#define USE_COLORED_OUTPUT //comment out this line to remove color from output
#ifdef USE_COLORED_OUTPUT
// text colors
    #define ANSI_COLOR_RED     "\x1b[31m"
    #define ANSI_COLOR_GREEN   "\x1b[32m"
    #define ANSI_COLOR_YELLOW  "\x1b[33m"
    #define ANSI_COLOR_BLUE    "\x1b[34m"
    #define ANSI_COLOR_MAGENTA "\x1b[35m"
    #define ANSI_COLOR_CYAN    "\x1b[36m"
    #define ANSI_COLOR_RESET   "\x1b[0m"
#else
    #define ANSI_COLOR_RED     ""
    #define ANSI_COLOR_GREEN   ""
    #define ANSI_COLOR_YELLOW  ""
    #define ANSI_COLOR_BLUE    ""
    #define ANSI_COLOR_MAGENTA ""
    #define ANSI_COLOR_CYAN    ""
    #define ANSI_COLOR_RESET   ""
#endif

#endif
