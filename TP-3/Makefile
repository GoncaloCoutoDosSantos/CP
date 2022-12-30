CC = nvcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
#CFLAGS = -std=c99 -O2 -ftree-vectorize -msse4 -funroll-loops -g -fno-omit-frame-pointer -fopt-info -fopenmp 
MAX = 256
CFLAGS = -std=c++11 -O3 -DMAX=$(MAX)
.DEFAULT_GOAL = k_means
THREADS = 40
CP_CLUSTERS = 4

k_means: $(SRC)k_means.cu $(BIN)utils.o
	$(CC) $(CFLAGS)  $(SRC)k_means.cu $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.cu $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.cu -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

profile:
	nvprof ./$(BIN)$(EXEC) 100000000 $(CP_CLUSTERS)