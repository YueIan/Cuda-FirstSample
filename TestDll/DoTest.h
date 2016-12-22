#pragma once
class __declspec(dllexport) DoTest
{
public:
    DoTest(void);
    ~DoTest(void);
};
extern "C"
{
    void __declspec(dllexport) __stdcall GetXXX() {
    }
}
