#include "crop_filter/core/asserts.h"
#include <cstdlib>

// required include for debug break in msvc compiler
#if _MSC_VER
#include <intrin.h>
#endif

void asserts_report_failure(logger& logger, const char* expression, const char* message, const char* file, u32 line) {
	LOGF(logger, "Assertion Failure: %s, message: '%s', in file: %s, line: %d", expression, message, file, line);
	
	// close program
	std::exit(-1);
}
