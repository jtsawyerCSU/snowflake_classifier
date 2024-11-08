#pragma once

#ifndef _logger_H_
#define _logger_H_

#include "crop_filter/core/defines.h"
#include <functional>

struct logger {
	
	// helper struct to support multiple callback types
	struct callback_t {
	public:
		callback_t(std::function<void (const char*)> callback);
		callback_t(std::function<void (const char*, size_t)> callback);
		void operator()(const char* message, size_t message_length);
	private:
		std::function<void (const char*, size_t)> m_callback;
	};
	
	// normal_callback is where normal (non error) logs will be sent
	// error_callback is where error logs will be sent
	logger(callback_t normal_callback, 
		   callback_t error_callback);
	
	enum log_level : u8 {
		Fatal,
		Error,
		Warn,
		Info,
		log_level_max
	};
	
	void log_output(log_level level, const char* message, ...);
	
	callback_t m_normal_callback;
	callback_t m_error_callback;
};

#define LOGF(log_instance, message, ...) log_instance.log_output(logger::log_level::Fatal, message, ##__VA_ARGS__)
#define LOGE(log_instance, message, ...) log_instance.log_output(logger::log_level::Error, message, ##__VA_ARGS__)
#define LOGW(log_instance, message, ...) log_instance.log_output(logger::log_level::Warn, message, ##__VA_ARGS__)
#define LOGI(log_instance, message, ...) log_instance.log_output(logger::log_level::Info, message, ##__VA_ARGS__)

#endif
