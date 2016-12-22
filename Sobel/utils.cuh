#include <fstream>

void MyDumpBuffer(const char* name, float* src, int count){
    std::ofstream of;
    of.open(name, std::ios::out);
    of.write((char*)src, count*sizeof(float));
    of.close();
}

void MyDumpBuffer(const char* name, short* src, int count){
    std::ofstream of;
    of.open(name, std::ios::out);
    of.write((char*)src, count*sizeof(short));
    of.close();
}