#ifndef   __INCLUDE_HIPPO_TPL_H
#define   __INCLUDE_HIPPO_TPL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


int write_uint8(FILE *fp, uint8_t value);
int read_uint8(uint8_t* buffer, uint8_t* value);
int write_uint32(FILE *fp, uint32_t value);
int read_uint32(uint8_t* buffer, uint32_t* value);
int write_uint8_array(FILE *fp, uint8_t *array, size_t n);
int read_uint8_array(uint8_t* buffer, uint8_t *array, size_t n);
void write_float(FILE *fp, float value);
int read_float(uint8_t* buffer, float* value);
void write_double(FILE *fp, double value);
int read_double(uint8_t* buffer, double* value);


#endif
