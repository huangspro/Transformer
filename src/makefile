# Makefile for your project using Armadillo

# Compiler
CXX = g++

# Compiler flags 
CXXFLAGS = -O2 -w -std=c++11

# Libraries
LIBS = -larmadillo

# Source files
SRCS = Attention.C Embedding.C main.C Normalization.C WordList.C

# Object files (replace .C with .o)
OBJS = $(SRCS:.C=.o)

# Executable name
TARGET = my_program

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Compile source files to object files
%.o: %.C
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)
