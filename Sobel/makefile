sobel: initialization.cu sobel2d.cu sobel3d.cu
	nvcc -shared -Xcompiler -fPIC -o sobel.so initialization.cu sobel2d.cu sobel3d.cu

testApp: testApp.o
	g++ testApp.o -o testApp -ldl

.PHONY: clean

clean:
	rm -f ./*.o ./*.so 