#include "crop_filter/core/logger.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <malloc.h>
#ifndef alloca
#include <alloca.h>
#endif

logger::logger(callback_t normal_callback, 
			   callback_t error_callback) : 
	m_normal_callback(normal_callback),
	m_error_callback(error_callback) {}

void logger::log_output(log_level level, const char* message, ...) {
	const char* level_identifier[log_level::log_level_max] = {"[FATAL]: ", "[ERROR]: ", "[WARN]:  ", "[INFO]:  "};
	size_t identifier_width = std::strlen(level_identifier[level]);
	size_t msg_length = 1024;
	
retry:
	
	char* formatted_message = static_cast<char*>(alloca(msg_length));
	std::memset(formatted_message, 0, msg_length);
	
	std::strncpy(formatted_message, level_identifier[level], identifier_width);
	
	std::va_list arg_ptr;
	va_start(arg_ptr, message);
	s32 count = std::vsnprintf(formatted_message + identifier_width, msg_length - identifier_width - 1, message, arg_ptr);
	va_end(arg_ptr);
	
	if (count < 0) {
		LOGE((*this), "Logger: Encoding error!\n\tEncoding: %s", message);
		return;
	} else if ((count >= (msg_length - identifier_width - 1)) && (count < (4096 - 1))) {
		LOGE((*this), "Logger: Maximum message length exceeded!");
		return;
	}
	
	// add a newline
	formatted_message[count] = '\n';
	
	// send log to different callbacks depending on error level
	if (level < log_level::Error) {
		m_normal_callback(formatted_message, (size_t)(count + 1));
	} else {
		m_error_callback(formatted_message, (size_t)(count + 1));
	}
}

logger::callback_t::callback_t(std::function<void (const char*)> callback) : 
	callback_t(
		[=](const char* message, size_t message_length) -> void {
			callback(message);
		}) {}

logger::callback_t::callback_t(std::function<void (const char*, size_t)> callback) : 
	m_callback(callback) {}

void logger::callback_t::operator()(const char* message, size_t message_length) {
	m_callback(message, message_length);
}
