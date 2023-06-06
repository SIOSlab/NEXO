f_files := $(wildcard *.f90)

all: nexo_lib nexo_py

nexo_lib: $(f_files)
	gfortran -c *.f90 -Wall -std=f2008 -g -O3
	ar rcs nexo.a *.o
	rm *.o

nexo_py: nexo.f90 nexo.a
	python3 -m numpy.f2py nexo.f90 -m nexo -h nexo.pyf
	python3 -m numpy.f2py -c nexo.pyf nexo.a -lblas -llapack
	rm -f nexo.pyf
