#ifndef ALT_THREE_KERNEL
#define ALT_THREE_KERNEL


#include <stdio.h>
#include "harness/sdaccel/CLKernel.h"

#include "defs.h"

class AltThreeKernel : public CLKernel {
    static const unsigned int GLOBAL_WORK_GROUP_SIZE = 1;
    static const unsigned int LOCAL_WORK_GROUP_SIZE  = 1;

    static size_t num_points_training;
    static size_t num_points_testing;
    static size_t num_features;

    FeatureType* data_points;
    LabelType* labels;
    FeatureType* parameter_vector;
public:
    AltThreeKernel(
        const char *filename,
        const char* kernel_name,
        cl_device_type device_type,
        FeatureType* data_points,
        LabelType* labels,
        FeatureType* parameter_vector);
    int setup_kernel ();
    int load_data();
    bool check_results();
    void clean_up();
};

#endif
