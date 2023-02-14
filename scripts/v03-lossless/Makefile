build: pyentropy.c pyentropy_setup.py
	gcc -O -fPIC --shared -ID:/Programs/Python38/include pyentropy.c -ID:/Programs/Python38/lib/site-packages/numpy/core/include -LD:/Programs/Python38/libs -o libpyentropy.dll -lpython3
#	python pyentropy_setup.py build_ext --inplace
	python pyentropy_setup.py build
	python pyentropy_setup.py install
