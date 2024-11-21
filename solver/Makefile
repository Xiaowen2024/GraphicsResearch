CC = g++
CFLAGS = -std=c++17 -O3 -Wall
TARGET = GeneralSolver
OBJECTS = main.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS) *.exe *~

.PHONY: clean
