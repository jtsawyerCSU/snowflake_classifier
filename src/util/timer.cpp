#include "util/timer.h"
#include <iostream>

Timer::Timer() {
	std::cout << "\n\n\n\n\n";
	std::cout << std::flush;
}

void Timer::lap() {
	std::chrono::time_point<std::chrono::high_resolution_clock> newTime = std::chrono::high_resolution_clock::now();
	latest = std::chrono::duration_cast<std::chrono::microseconds>(newTime - previousTime).count();
	
	if (latest < best) {
		best = latest;
	}
	if (latest > worst) {
		worst = latest;
	}
	
	static int iteration = 0;
	
	average = (average * iteration + latest) / (++iteration);

	std::cout << "Iteration: " << iteration << '\n';

	print();
	
	previousTime = newTime;
}

void Timer::discardTime() {
	previousTime = std::chrono::high_resolution_clock::now();
}
	
void Timer::print() {
	std::cout << "\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A\r";
	std::cout << "Average: " << average << "μs    \n";
	std::cout << "Best:    " << best << "μs    \n";
	std::cout << "Worst:   " << worst << "μs    \n";
	std::cout << "Latest:  " << latest << "μs    \n";
	//std::cout << std::flush;
}
