// TestApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <windows.h>
#include <fstream>
#include <vector>

typedef int (__stdcall *GetContour)(int, int*, int*, int*);

typedef int (__stdcall *Get2DBorder)(float*, float*, int, int);

typedef int (__stdcall *Get3DBorder)(short* , short* , const int , const int , const int);

void MyDumpBuffer(std::wstring name, float* src, int count){
    std::ofstream of;
    of.open(name, std::ios::out);
    of.write((char*)src, count*sizeof(float));
    of.close();
}

void MyDumpBuffer(std::wstring name, short* src, int count){
    std::ofstream of;
    of.open(name, std::ios::out|std::ios::binary);
    of.write((char*)src, count*sizeof(short));
    of.close();
}

void MyDumpBuffer(std::wstring name, char* src, int count){
    std::ofstream of;
    of.open(name, std::ios::out|std::ios::binary);
    of.write((char*)src, count*sizeof(char));
    of.close();
}

void MyDumpBufferStream(std::wstring name, char* src, int count){
    std::ofstream of;
    of.open(name, std::ios::out);
    of.write((char*)src, count*sizeof(char));
    of.close();
}

void MyReadFile(std::wstring name, float* src, int count){
    std::ifstream inf;
    inf.open(name, std::ios::in);
    inf.read((char*)src, count*sizeof(float));
    inf.close();
}

void MyReadFile(std::wstring name, short* src, int count){
    std::ifstream inf;
    inf.open(name, std::ios::in|std::ios::binary);
    inf.read((char*)src, count*sizeof(short));
    inf.close();
}

void MyReadFile(std::wstring name, char* src, int count){
    std::ifstream inf;
    inf.open(name, std::ios::in|std::ios::binary);
    inf.read((char*)src, count*sizeof(char));
    inf.close();
}

int Test2DSobel();

int Test3DSobel();

int _tmain(int argc, _TCHAR* argv[])
{
    return Test3DSobel();

}

int Test2DSobel(
) {
    HINSTANCE hInstLibrary =LoadLibrary(L"Sobel.dll");

    Get2DBorder _get_2d_border = (Get2DBorder)GetProcAddress(hInstLibrary, "Get2DBorder");

    const int m = 512;
    const int n = 512;
    const int array_size = m*n;
    std::vector<float> _src(array_size);
    std::vector<float> _des(array_size);
    float* src = &_src[0];
    float* des = &_des[0];
    memset(src, 0, array_size*sizeof(float));
    memset(des, 0, array_size*sizeof(float));

    /*for (int i=64;i<192;i++) {
    for (int j=64;j<192;j++) {
    src[i*m + j] = 100;
    }
    }*/

    MyReadFile(L"398.raw", src, array_size);
    //MyDumpBuffer(L"E:\\workspace\\NVIDIA\\FirstSample\\x64\\Debug\\in", src, array_size);

    _get_2d_border(src, des, m, n);

    MyDumpBuffer(L"E:\\workspace\\NVIDIA\\FirstSample\\x64\\Debug\\in", src, array_size);
    MyDumpBuffer(L"E:\\workspace\\NVIDIA\\FirstSample\\x64\\Debug\\out", des, array_size);

    FreeLibrary(hInstLibrary);
    return 0;
}

int Test3DSobel(
) {
    unsigned int n = 512*512*400*4;
    char* s1 = new char[n+27];
    MyReadFile(L"D:\\Raw\\1.2.392.200036.9116.4.2.6953.1689.4003.raw", s1, n+27);
    MyDumpBuffer(L"E:\\c_in_3d_o", (s1+27), n);


    /*unsigned int n = 348;//181*217*181*2;
    char* s1 = new char[n];
    MyReadFile(L"E:\\Precision Medicine\\Adult27-55-e5450fc\\Adt27-55_02\\Adt27-55_02_FullLabels.hdr", s1, n);
    printf("%d", s1);
    MyDumpBufferStream(L"E:\\xxx", s1, n);*/
    
    HINSTANCE hInstLibrary =LoadLibrary(L"Sobel.dll");

    Get3DBorder _get_3d_border = (Get3DBorder)GetProcAddress(hInstLibrary, "Get3DBorder");

    const int x = 256;
    const int y = 256;
    const int z = 35;
    const int array_size = x*y*z;
    std::vector<short> _src(array_size);
    std::vector<short> _des(array_size);
    short* src = &_src[0];
    short* des = &_des[0];
    memset(src, 0, array_size*sizeof(short));
    memset(des, 0, array_size*sizeof(short));

    /*for (int i=0;i<z;i++) {
        for (int j=0;j<y;j++) {
            for (int k=0;k<x;k++) {
                if (sqrtf(powf((float)(i-50), 2) + powf((float)(j-256), 2) + powf((float)(k-256), 2)) < 30.f) {
                    src[i*y*x + j*x + k] = 100;
                }
            }
        }
    }*/

    MyReadFile(L"E:\\bbb", src, array_size);
    //MyDumpBuffer(L"E:\\workspace\\NVIDIA\\FirstSample\\x64\\Debug\\in", src, array_size);

    _get_3d_border(src, des, x, y,z);
    
    MyDumpBuffer(L"E:\\c_in_3d", src, array_size);
    MyDumpBuffer(L"E:\\c_out_3d", des, array_size);

    FreeLibrary(hInstLibrary);
    return 0;
}

