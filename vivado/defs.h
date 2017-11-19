#ifndef SPAMFILTER_DEFS
#define SPAMFILTER_DEFS

#include <string>
#include <map>

// constants and default values for some configuration parameters
#define NUM_FEATURES      1024
#define NUM_SAMPLES       100
#define DATA_SET_SIZE     102400
#define NUM_TRAINING      90
#define NUM_TESTING       10
#define NUM_VALIDATION    0
#define LAMBDA            0 //regularization parameter
#define STEP_SIZE         50 //step size (eta)
#define NUM_EPOCHS        1
#define MAX_NUM_EPOCHS    1
#define NUM_RUNS          1
#define ERROR_GOAL        0.001
#define CHARACTERISTIC_TIME (NUM_EPOCHS * NUM_TRAINING / 3)

#define POSITIVE_LABEL    1
#define NEGATIVE_LABEL    0


// type definitions
typedef float FeatureType;
typedef float LabelType;
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
