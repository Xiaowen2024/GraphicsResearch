# Makefile for compiling
CXX = c++
CXXFLAGS = -std=c++17 -O3 -pedantic -Wall

# Source files
SRCS = GeneralSolver.cpp WalkOnStars.cpp
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = solver

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
