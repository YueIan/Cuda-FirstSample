# Cuda-FirstSample
Do border by Sobel using cuda GPU

# Interface: 
2D Border Extraction:
```bash
@param  src(float*)  input buffer
@param  des(float*)  output border buffer
@param  m(int)  input buffer's width
@param  n(int)  input buffer's height
int Get2DBorder(float* src, float* des, const int m , const int n)
```
3D Border Extraction:
```bash
@param  param_src(short*)  input buffer
@param  param_des(short*)  output border buffer
@param  x(int)  input buffer's width
@param  y(int)  input buffer's height
@param  z(int)  input buffer's height
int Get3DBorder(short* param_src, short* param_des, const int x , const int y, const int z) 
```
