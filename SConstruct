SRC = Split('''svm_common.c
                svm_learn.c
                svm_hideo.c
                svm-light.cpp''')

env = Environment(CXX='g++',CC='g++', CFLAGS = '-O3', LINKFLAGS = '-fopenmp')
env.StaticLibrary(target = 'libsvm-light', source = SRC)
