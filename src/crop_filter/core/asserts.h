#pragma once

#ifndef _asserts_H_
#define _asserts_H_

#include "crop_filter/core/logger.h"

void asserts_report_failure(logger& logger, const char* expression, const char* message, const char* file, u32 line);

#define ASSERT(logger_instance, expression)                                           \
{                                                                                     \
	if (expression) {                                                                 \
	} else {                                                                          \
		asserts_report_failure(logger_instance, #expression, "", __FILE__, __LINE__); \
	}                                                                                 \
}

#define ASSERT_MSG(logger_instance, expression, message)                                   \
{                                                                                          \
	if (expression) {                                                                      \
	} else {                                                                               \
		asserts_report_failure(logger_instance, #expression, message, __FILE__, __LINE__); \
	}                                                                                      \
}

#endif
