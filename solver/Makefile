# Example Makefile
CC = g++
CFLAGS = -std=c++17 -O3 -Wall
TARGET = GeneralSolver.o

all: $(TARGET)

$(TARGET): main.cpp
	$(CC) $(CFLAGS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean
