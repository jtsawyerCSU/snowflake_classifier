#pragma once

#ifndef _defines_H_
#define _defines_H_

#include <cstddef>
#include <ctime>

// unsigned int types
typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

// signed int types
typedef signed char      s8;
typedef signed short     s16;
typedef signed int       s32;
typedef signed long long s64;

// floating point types
typedef float  f32;
typedef double f64;

// compiler specific static assertions
#if defined(__gcc__) || defined(__clang__)
#define STATIC_ASSERT _Static_assert
#else
#define STATIC_ASSERT static_assert
#endif

// ensure types match expected sizes
STATIC_ASSERT(sizeof(u8)  == 1, "Expected u8  to be 1 byte.");
STATIC_ASSERT(sizeof(u16) == 2, "Expected u16 to be 2 bytes.");
STATIC_ASSERT(sizeof(u32) == 4, "Expected u32 to be 4 bytes.");
STATIC_ASSERT(sizeof(u64) == 8, "Expected u64 to be 8 bytes.");

STATIC_ASSERT(sizeof(s8)  == 1, "Expected s8  to be 1 byte.");
STATIC_ASSERT(sizeof(s16) == 2, "Expected s16 to be 2 bytes.");
STATIC_ASSERT(sizeof(s32) == 4, "Expected s32 to be 4 bytes.");
STATIC_ASSERT(sizeof(s64) == 8, "Expected s64 to be 8 bytes.");

STATIC_ASSERT(sizeof(f32) == 4, "Expected f32 to be 4 bytes.");
STATIC_ASSERT(sizeof(f64) == 8, "Expected f64 to be 8 bytes.");

#endif
