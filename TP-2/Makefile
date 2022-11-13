CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
CFLAGS = -std=c99 -O2 -ftree-vectorize -msse4 -funroll-loops -g -fno-omit-frame-pointer -fopt-info -fopenmp 
#CFLAGS = -std=c99 -O3 -fopenmp -fopt-info #-DTEST
.DEFAULT_GOAL = k_means
THREADS = 10

k_means: $(SRC)k_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

runseq:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) 1

run:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)