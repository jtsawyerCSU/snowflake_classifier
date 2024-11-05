#pragma once

#ifndef _logger_H_
#define _logger_H_

#include "crop_filter/core/defines.h"
#include <utility>

struct logger {
public:
	
	logger(std::function<void (const char*, size_t)> callback);
	
	enum log_level : u8 {
		Fatal,
		Error,
		Warn,
		Info,
		log_level_max
	};
	
	void log_output(log_level level, const char* message, ...);
	
private:
	std::function<void (const char*, size_t)> m_callback;
};

#define LOGF(log_instance, message, ...) log_instance.log_output(log_level::Fatal, message, ##__VA_ARGS__)
#define LOGE(log_instance, message, ...) log_instance.log_output(log_level::Error, message, ##__VA_ARGS__)
#define LOGW(log_instance, message, ...) log_instance.log_output(log_level::Warn, message, ##__VA_ARGS__)
#define LOGI(log_instance, message, ...) log_instance.log_output(log_level::Info, message, ##__VA_ARGS__)

#endif
