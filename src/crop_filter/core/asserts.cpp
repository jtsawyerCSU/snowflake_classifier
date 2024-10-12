#include "crop_filter/core/asserts.h"
#include "crop_filter/core/logger.h"

// required include for debug break in msvc compiler
#if _MSC_VER
#include <intrin.h>
#endif

static void breakpoint() {

#if _MSC_VER
	__debugbreak();
#else
	__builtin_trap();
#endif

}

void asserts_report_failure(const char* expression, const char* message, const char* file, u32 line) {
	LOGF("Assertion Failure: %s, message: '%s', in file: %s, line: %d", expression, message, file, line);
}
