#pragma once

#ifndef _logger_H_
#define _logger_H_

#include "crop_filter/core/defines.h"

enum log_level : u8 {
	Fatal,
	Error,
	Warn,
	Info,
	log_level_max
};

void log_output(log_level level, const char* message, ...);

#define LOGF(message, ...) log_output(log_level::Fatal, message, ##__VA_ARGS__)
#define LOGE(message, ...) log_output(log_level::Error, message, ##__VA_ARGS__)
#define LOGW(message, ...) log_output(log_level::Warn, message, ##__VA_ARGS__)
#define LOGI(message, ...) log_output(log_level::Info, message, ##__VA_ARGS__)

#endif
