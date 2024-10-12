#pragma once

#ifndef _asserts_H_
#define _asserts_H_

#include "crop_filter/core/defines.h"

void asserts_report_failure(const char* expression, const char* message, const char* file, u32 line);

#define ASSERT(expression)                                           \
{                                                                    \
	if (expression) {                                                \
	} else {                                                         \
		asserts_report_failure(#expression, "", __FILE__, __LINE__); \
	}                                                                \
}

#define ASSERT_MSG(expression, message)                                   \
{                                                                         \
	if (expression) {                                                     \
	} else {                                                              \
		asserts_report_failure(#expression, message, __FILE__, __LINE__); \
	}                                                                     \
}

#endif
