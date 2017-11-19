//
//  CLMemObj.cpp
//  OpenCLHarness
//
//  Created by Tim on 12/18/15.
//  Copyright (c) 2015 Tim. All rights reserved.
//


#include "CLMemObj.h"

CLMemObj::CLMemObj() {
    this->mem_data = 0;
    this->elt_size = 0;
    this->length = 0;
    this->flags = 0;
}

CLMemObj::CLMemObj(void *mem_data, int elt_size, int length, cl_mem_flags flags) {
    // Initialize the data info constants
    this->mem_data = mem_data;
    this->elt_size = elt_size;
    this->length = length;
    this->flags = flags;
}

void * CLMemObj::get_data() {
    return mem_data;
}

int CLMemObj::get_element_size() {
    return elt_size;
}

int CLMemObj::get_length() {
    return length;
}

cl_mem_flags CLMemObj::get_flags() {
    return flags;
}
