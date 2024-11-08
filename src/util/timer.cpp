#include "util/timer.h"
#include <iostream>

timer::timer() {}

void timer::lap() {
	std::chrono::time_point<std::chrono::high_resolution_clock> newTime = std::chrono::high_resolution_clock::now();
	latest = std::chrono::duration_cast<std::chrono::microseconds>(newTime - previousTime).count();
	
	if (latest < best) {
		best = latest;
	}
	if (latest > worst) {
		worst = latest;
	}
	
	average = (average * iteration + latest) / (++iteration);

	print();
	
	previousTime = newTime;
}

void timer::discardTime() {
	previousTime = std::chrono::high_resolution_clock::now();
}

void timer::print() {
	std::cout << "Iteration: " << iteration << '\n';
	std::cout << "Average: " << average << "μs\n";
	std::cout << "Best:    " << best << "μs\n";
	std::cout << "Worst:   " << worst << "μs\n";
	std::cout << "Latest:  " << latest << "μs\n";
}

void timer::reset_position() {
	std::cout << "\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A\r";
}
