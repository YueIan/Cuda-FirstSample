// dllmain.h : Declaration of module class.

class CTestDllModule : public ATL::CAtlDllModuleT< CTestDllModule >
{
public :
	DECLARE_LIBID(LIBID_TestDllLib)
	DECLARE_REGISTRY_APPID_RESOURCEID(IDR_TESTDLL, "{0F58BD56-7C21-4E3D-B696-9AA93D4C2BB8}")
};

extern class CTestDllModule _AtlModule;
