#pragma once

#ifndef _Timer_H
#define _Timer_H

#include <chrono>
#include <cstdint>

struct Timer {

	Timer();
	
	void lap();

	void discardTime();
	
	void print();
	
	// all times are in microseconds
	uint64_t average{}, worst{}, best{std::numeric_limits<uint64_t>::max()}, latest{};
	std::chrono::time_point<std::chrono::high_resolution_clock> previousTime{std::chrono::high_resolution_clock::now()};
	
};

#endif
