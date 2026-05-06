CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic
TARGET ?= parallel_ap
SRC := src/sparse_ap_omp.cpp

.PHONY: all serial clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

serial:
	$(CXX) -std=c++17 -O3 -Wall -Wextra -pedantic $(SRC) -o $(TARGET)_serial

clean:
	rm -f $(TARGET) $(TARGET).exe $(TARGET)_serial labels.csv src/*.o
