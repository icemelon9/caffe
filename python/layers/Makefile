all:
	python setup.py build_ext --inplace
	rm -rf build

clean:
	rm -f *.pyc */*.pyc nms/*.so nms/cpu_nms.c nms/gpu_nms.cpp

.PHONY: clean
