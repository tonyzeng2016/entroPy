 #include "tpl.h"
 int write_uint8(FILE *fp, uint8_t value)
{
    return fwrite(&value, sizeof(value), 1, fp) == 1 ? 0 : 1;
/***
int fwrite( const void *buffer, size_t size, size_t count, FILE *stream );
将buffer中的count个大小为size的对象写入stream,返回值为完成写入的个数
***/
}

 int read_uint8(uint8_t* buffer, uint8_t* value)
{
    *value = *buffer;
    return sizeof(*value);
}

 int write_uint32(FILE *fp, uint32_t value)
{
    uint8_t buffer[4];
    buffer[0] = (uint8_t)(value & 0xFF);
    buffer[1] = (uint8_t)(value >> 8);
    buffer[2] = (uint8_t)(value >> 16);
    buffer[3] = (uint8_t)(value >> 24);
    return fwrite(buffer, sizeof(uint8_t), 4, fp) == 4 ? 0 : 1;
}

 int read_uint32(uint8_t* buffer, uint32_t* value)
{
    *value  = ((uint32_t)buffer[0]);
    *value |= ((uint32_t)buffer[1] << 8);
    *value |= ((uint32_t)buffer[2] << 16);
    *value |= ((uint32_t)buffer[3] << 24);
    return sizeof(*value);
}

 int write_uint8_array(FILE *fp, uint8_t *array, size_t n)
{
    size_t i;
    int ret = 0;
    for (i = 0;i < n;++i) {
        ret |= write_uint8(fp, array[i]);
    }
    return ret;
}

 int read_uint8_array(uint8_t* buffer, uint8_t *array, size_t n)
{
    size_t i;
    int ret = 0;
    for (i = 0;i < n;++i) {
        int size = read_uint8(buffer, &array[i]);
        buffer += size;
        ret += size;
    }
    return ret;
}

 void write_float(FILE *fp, float value)
{
    /*
        We assume:
            - sizeof(floatval_t) = sizeof(double) = sizeof(uint64_t)
            - the byte order of floatval_t and uint64_t is the same
            - ARM's mixed-endian is not supported
    */
    uint32_t iv;
    uint8_t buffer[4];

    /* Copy the memory image of floatval_t value to uint64_t. */
    memcpy(&iv, &value, sizeof(iv));

    buffer[0] = (uint8_t)(iv & 0xFF);
    buffer[1] = (uint8_t)(iv >> 8);
    buffer[2] = (uint8_t)(iv >> 16);
    buffer[3] = (uint8_t)(iv >> 24);
    fwrite(buffer, sizeof(uint8_t), 4, fp);
}

 int read_float(uint8_t* buffer, float* value)
{
    uint32_t iv;
    iv  = ((uint32_t)buffer[0]);
    iv |= ((uint32_t)buffer[1] << 8);
    iv |= ((uint32_t)buffer[2] << 16);
    iv |= ((uint32_t)buffer[3] << 24);
    memcpy(value, &iv, sizeof(*value));
    return sizeof(*value);
}
void write_double(FILE *fp, double value)
{
    /*
        We assume:
            - sizeof(floatval_t) = sizeof(double) = sizeof(uint64_t)
            - the byte order of floatval_t and uint64_t is the same
            - ARM's mixed-endian is not supported
    */
    uint64_t iv;
    uint8_t buffer[8];

    /* Copy the memory image of floatval_t value to uint64_t. */
    memcpy(&iv, &value, sizeof(iv));

    buffer[0] = (uint8_t)(iv & 0xFF);
    buffer[1] = (uint8_t)(iv >> 8);
    buffer[2] = (uint8_t)(iv >> 16);
    buffer[3] = (uint8_t)(iv >> 24);
    buffer[4] = (uint8_t)(iv >> 32);
    buffer[5] = (uint8_t)(iv >> 40);
    buffer[6] = (uint8_t)(iv >> 48);
    buffer[7] = (uint8_t)(iv >> 56);
    fwrite(buffer, sizeof(uint8_t), 8, fp);
}

 int read_double(uint8_t* buffer, double* value)
{
    uint64_t iv;
    iv  = ((uint64_t)buffer[0]);
    iv |= ((uint64_t)buffer[1] << 8);
    iv |= ((uint64_t)buffer[2] << 16);
    iv |= ((uint64_t)buffer[3] << 24);
    iv |= ((uint64_t)buffer[4] << 32);
    iv |= ((uint64_t)buffer[5] << 40);
    iv |= ((uint64_t)buffer[6] << 48);
    iv |= ((uint64_t)buffer[7] << 56);
    memcpy(value, &iv, sizeof(*value));
    return sizeof(*value);
}
