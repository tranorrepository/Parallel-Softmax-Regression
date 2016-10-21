#ifndef MNIST_UTILS
#define MNIST_UTILS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <string>
#include <math.h>
#include <vector>

#include "mnist_defs.h"

using namespace std;

/******************************************************************************/
/*                                                                            */
/* General Utilities                                                          */
/*                                                                            */
/******************************************************************************/

//macro for min; this is prone to double eval,
// use only with values, not function calls
template <typename Type>
Type min_val(Type a, Type b) {
    return (a < b) ? a : b;
}

//find maximum value in a vector
template <typename Type>
Type max_val(vector<Type> v) {
    Type max = v[0];
    for(int i=0; i<v.size();i++){
        if(v[i]>max){
            max=v[i];
        }
    }

    return max;
}

//find sum in a vector
template <typename Type>
Type sum_val(vector<Type> v) {
    Type sum = 0;
    for(int i=0; i<v.size();i++){
        sum+=v[i];
    }

    return sum;
}

// swap elements in an array
template <typename Type>
void swap(
    Type *array,
    size_t index1,
    size_t index2,
    const size_t elements_per_chunk = 1) {

    Type* temp = new Type[elements_per_chunk];
    index1 *= elements_per_chunk;
    index2 *= elements_per_chunk;
    const size_t size_of_chunk = elements_per_chunk * sizeof(Type);
    memcpy(temp, &array[index1], size_of_chunk);
    memcpy(&array[index1], &array[index2], size_of_chunk);
    memcpy(&array[index2], temp, size_of_chunk);
    delete[] temp;
}

// shuffle array randomly
template <typename KeyType, typename ValueType>
void shuffleKeyValue(
    KeyType *key_array,
    ValueType *value_array,
    size_t size,
    const size_t key_elements_per_chunk = 1,
    const size_t value_elements_per_chunk = 1) {

    //can't use % 0
    if (size == 0)
        return;

    srand(static_cast<unsigned int>(time(0)));

    for (size_t i = 0; i < size; i++) {
        size_t j = rand() % size;
        swap(key_array, i, j, key_elements_per_chunk);
        swap(value_array, i, j, value_elements_per_chunk);
    }
}


// initialize default configuration options
TrainingOptions initDefaultTrainingOptions(double* step_size);
BenchmarkOptions initDefaultBenchmarkOptions();


// parses training- and benchmark- related command line arguments
void parse_command_line_args(
    int argc,
    char** argv,
    TrainingOptions* training_options = NULL,
    BenchmarkOptions* benchmark_options = NULL,
    std::string* path_to_data = NULL);




/******************************************************************************/
/*                                                                            */
/* Logistic and softmax Regression-Specific Utilities                                     */
/*                                                                            */
/******************************************************************************/

// computes logistic function for a given double
double logisticFunction(double exponent);
// computes logistic function for a given parameter vector (parameter_vector)
// and a data point (data_point_i)
std::vector<double>  softmaxFunction(
    FeatureType* theta,
    FeatureType* x_i,
    const size_t num_features);

std::vector<float> softmaxFunctionFloat(
    FeatureType* parameter_vector,
    FeatureType* data_point_i,
    const size_t num_features );

double logisticFunction(
    FeatureType* theta,
    FeatureType* x_i,
    const size_t num_features);

int getLabelIndex(vector<double> posibiility_each);



// computes true positive rate, false positive rate and error rate, then adds
// these values to variables passed through pointers
double computeErrorRate(
    DataSet data_set,
    Dictionary* errors = NULL);

LabelType getSoftmaxPrediction(
    FeatureType* parameter_vector,
    FeatureType* data_point_i,
    size_t num_features
    );  


double computeSoftmaxErrorRate(
    DataSet data_set,
    Dictionary* errors = NULL);

// loss function for the whole batch (Negative LogLikelihood for
// Bernoulli probability distribution)
double lossFunction(
    DataSet data_set);

double softmaxLossFunction(DataSet data_set);

// Benchmark method for the logistic regression model.
// It takes a training function, a data set and some configuration options,
// and then calls the training function several times until the desired
// accuracy is reached or the number of epochs exceeds the given limit.
// The training time is measured for multiple runs and the results are averaged.
void trainAndTest(
    void (*training_function)(DataSet, TrainingOptions),
    const char* name,
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options);

// runs the training n epochs at a time
//until it reaches m epochs, printing error
//rate every n epochs 
void convergenceRate(
    void (*training_function)(DataSet, TrainingOptions),
    const char* name,
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options);

//run the training one epoch at a time until some accurancy
//goal is reached or reach maximum epochs
void convergenceTime(
    void (*training_function)(DataSet, TrainingOptions),
    const char* name,
    DataSet data_set,
    TrainingOptions training_options,
    BenchmarkOptions benchmark_options);

// determines whether a key already exists in a Dictionary
bool fieldExists(Dictionary dict, const char* key);

/******************************************************************************/
/*                                                                            */
/* Linear Algebra                                                             */
/*                                                                            */
/******************************************************************************/

#ifdef USE_OPENBLAS // ************************ using OpenBLAS *****************

//declarations for different functions

// adds two vectors and stores the results in the first one
void addVectors(
    float* a,
    float* b,
    const size_t size,
    const float scale_for_b = 1);

// computes dot product for two given vectors a and b
float dotProduct(
    float* a,
    float* b,
    const size_t size);

// computes norm 2 of a given vector v
float norm2(
    float* v,
    const size_t size);

// matrix-vector multiplication wrapper
void matrixVectorMultiply(
    float* A,
    float* v,
    float alpha,
    size_t num_points,
    size_t num_features,
    float* result);

// matrix-matrix multiplication wrapper
void matrixMatrixMultiply(
    float* matrixA,
    float* matrixB,
    float scalar,
    size_t num_data_points,
    size_t num_features,
    float* result);

// updates the parameters (parameter_vector)
void updateParameters(
    FeatureType* theta,
    FeatureType* gradient,
    size_t num_features,
    double step_size,
    bool revert = false );


#else // ************************ not using OpenBLAS ***************************

// adds two vectors and stores the results in the first one
template <typename Type1, typename Type2, typename Type3>
void addVectors(
    Type1* a,
    Type2* b,
    const size_t size,
    const Type3 scale_for_b) {

    for (size_t j = 0; j < size; j++)
        a[j] += (Type1) (scale_for_b * b[j]);

    // cblas_saxpy(size, scale_for_b, b, 1, a, 1);
}

// computes dot product for two given vectors a and b
template <typename Type1, typename Type2>
Type1 dotProduct(
    Type1* a,
    Type2* b,
    const size_t size) {

    Type1 result = 0;
    for (size_t j = 0; j < size; j++)
        result += a[j] * (Type1)b[j];

    return result;
    // return cblas_sdot(size, a, 1, b, 1);
}

// matrix-vector multiplication wrapper
template <typename Type1, typename Type2, typename Type3, typename Type4>
void matrixVectorMultiply(
    Type1* matrix,
    Type2* vect,
    Type3 scalar,
    size_t num_data_points,
    size_t num_features,
    Type4* result) {

    memset(result, 0, num_features * sizeof(Type4));

    for (size_t i = 0; i < num_data_points; i++)
        addVectors(
            result,
            &matrix[i * num_features],
            num_features,
            scalar * vect[i]);

    // cblas_sgemv(CblasRowMajor, CblasTrans, num_data_points, num_features,
    // scalar, matrix, num_features, vect, 1, 0, result, 1);
}

// computes norm 2 of a given vector v
template <typename Type>
float norm2(Type* v, const size_t size) {
    return sqrt(dotProduct(v, v, size));
}

// updates the parameters (parameter_vector)
template <typename Type>
void updateParameters(
    Type* parameter_vector,
    Type* gradient,
    size_t num_features,
    double step_size,
    bool revert = false) {

    double sign = revert ? 1 : -1;
    step_size *= sign;
    addVectors(parameter_vector, gradient, num_features, step_size);

    // cblas_saxpy (num_features, step_size, gradient, 1, parameter_vector, 1);
}

#endif // ************************ end of OpenBLAS conditional *****************


#endif
