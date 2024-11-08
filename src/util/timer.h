#pragma once

#ifndef _timer_H
#define _timer_H

#include <chrono>
#include <cstdint>

struct timer {

	timer();
	
	void lap();

	void discardTime();
	
	void print();
	
	// moves the terminal cursor back up
	// used to overwrite printed stats from the last iteration
	void reset_position();
	
	// all times are in microseconds
	uint64_t average{}, worst{}, best{std::numeric_limits<uint64_t>::max()}, latest{};
	std::chrono::time_point<std::chrono::high_resolution_clock> previousTime{std::chrono::high_resolution_clock::now()};
	int iteration = 0;
	
};

#endif
