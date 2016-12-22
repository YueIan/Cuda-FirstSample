# Cuda-FirstSample
Do border by Sobel using cuda GPU

# Interface: 
2D Border Extraction:
```bash
@param  src(T*)  input buffer
@param  des(T*)  output border buffer
@param  m(int)  input buffer's width
@param  n(int)  input buffer's height
int Get2DBorder(T* src, T* des, const int m , const int n)
T mean float or int
```
3D Border Extraction:
```bash
@param  param_src(T*)  input buffer
@param  param_des(T*)  output border buffer
@param  x(int)  input buffer's width
@param  y(int)  input buffer's height
@param  z(int)  input buffer's height
int Get3DBorder(T* param_src, T* param_des, const int x , const int y, const int z) 
T mean short or int
```
