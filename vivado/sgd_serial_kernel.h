#ifndef SGD_SERIAL_KERNEL
#define SGD_SERIAL_KERNEL

#include "defs.h"

// Top-level Kernel
// Third alternative version of Stochastic Gradient Descent
void SgdLR(FeatureType* global_data_points, LabelType* global_labels, FeatureType* global_parameter_vector);

#endif
