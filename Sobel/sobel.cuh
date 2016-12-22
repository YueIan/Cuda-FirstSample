#include "common.cuh"

extern "C"
{
    int MY_EXPORT Get2DBorder(float* src, float* des, const int m , const int n);

    int MY_EXPORT Get3DBorder(short* param_src, short* param_des, const int x , const int y, const int z);
}