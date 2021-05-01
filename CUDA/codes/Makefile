###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -arch=compute_61 -rdc=true -lcudadevrt
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L $(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I $(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart



##########################################################

## Make variables ##
SRC = src
# Target executable name:
EXE = try

# Object files:
OBJS = utils.o main.o
##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
%.o : $(SRC)/%.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
%.o : $(SRC)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
%.o : $(SRC)/%.cu $(SRC)/%.cuh
	$(NVCC)  $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)
%.o : $(SRC)/%.cu
	$(NVCC)  $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) *.o $(EXE)