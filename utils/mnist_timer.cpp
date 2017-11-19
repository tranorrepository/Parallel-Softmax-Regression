#include <stdio.h>
#include "mnist_timer.h"

// time in miliseconds

Timer::Timer(const char* given_name) : name(given_name) {
    total_time = 0;
    is_running = false;
}

Timer::~Timer() {
}

void Timer::start() {
    total_time = 0;
    Timer::resume();
}

void Timer::pause() {
    if (is_running) {
        is_running = false;
        total_time += (clock() - start_time) * 1000.0 / (double) CLOCKS_PER_SEC;
    }
}

void Timer::resume() {
    if (!is_running) {
        start_time = clock();
        is_running = true;
    }
}

double Timer::stop() {
    Timer::pause();
    return total_time;
}
