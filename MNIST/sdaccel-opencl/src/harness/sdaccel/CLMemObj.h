//
//  CLMemObj.h
//  OpenCLHarness
//
//  Created by Tim on 12/18/15.
//  Copyright (c) 2015 Tim. All rights reserved.
//

#ifndef __OpenCLHarness__CLMemObj__
#define __OpenCLHarness__CLMemObj__

#include <stdio.h>
#include <CL/cl.h>

class CLMemObj {
public:
    CLMemObj ();
    CLMemObj (void *, int , int, cl_mem_flags);
    void * get_data();
    int get_element_size();
    int get_length();
    cl_mem_flags get_flags();
private:
    void *mem_data;
    int elt_size;
    int length;
    cl_mem_flags flags;
    int err;
};

#endif /* defined(__OpenCLHarness__CLMemObj__) */
