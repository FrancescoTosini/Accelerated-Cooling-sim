# Location of the CUDA Toolkit
# CUDA_PATH?=/opt/cuda

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif

G_ARCH := 70
LOC_ARCH := 50

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
	$(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
	ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
		HOST_COMPILER ?= clang++
	endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
		ifeq ($(TARGET_OS),linux)
			HOST_COMPILER ?= arm-linux-gnueabihf-g++
		else ifeq ($(TARGET_OS),qnx)
			ifeq ($(QNX_HOST),)
				$(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
			endif
			ifeq ($(QNX_TARGET),)
				$(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
			endif
			export QNX_HOST
			export QNX_TARGET
			HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
		else ifeq ($(TARGET_OS),android)
			HOST_COMPILER ?= arm-linux-androideabi-g++
		endif
	else ifeq ($(TARGET_ARCH),aarch64)
		ifeq ($(TARGET_OS), linux)
			HOST_COMPILER ?= aarch64-linux-gnu-g++
		else ifeq ($(TARGET_OS), android)
			HOST_COMPILER ?= aarch64-linux-android-g++
		endif
	else ifeq ($(TARGET_ARCH),ppc64le)
		HOST_COMPILER ?= powerpc64le-linux-gnu-g++
	endif
endif
HOST_COMPILER ?= g++
NVCC          := nvcc -ccbin $(HOST_COMPILER)
# NVCCFLAGS     := -m${TARGET_SIZE}

# Debug build flags
ifeq ($(dbg),1)
	  NVCCFLAGS += -g -G
	  BUILD_TYPE := debug
else
	  BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Final binary
BIN = cooling
# Put all auto generated stuff to this build dir.
BUILD_DIR = ./build

# List of all source files.
SRCS = main.cu $(wildcard ./utils/*.cu)

# All .o files go to build dir.
OBJ = $(SRCS:%.cu=$(BUILD_DIR)/%.o)
# Gcc/Clang will create these .d files containing dependencies.
# keeping them will track header changes (i think :)
DEP = $(OBJ:%.o=%.d)

# extra headers
INCLUDES  := -I./utils

# where are we?
ifdef SCRATCH
    # compiling for galileo...
	GENCODE_FLAGS += -gencode arch=compute_$(G_ARCH),code=sm_$(G_ARCH)
else
    # compiling for local...
	GENCODE_FLAGS += -gencode arch=compute_$(LOC_ARCH),code=sm_$(LOC_ARCH)
endif

LIBRARIES += -lcusolver -lcublas -lcusparse

# Target rules
all: $(BIN)

# Default target named after the binary.
$(BIN) : $(BUILD_DIR)/$(BIN)

# Actual target of the binary - depends on all .o files.
$(BUILD_DIR)/$(BIN) : $(OBJ)
    # Create build directories - same structure as sources.
	mkdir -p $(@D)
    # Just link all the object files.
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ -o $@ $(LIBRARIES) 

# Include all .d files
-include $(DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cu
	mkdir -p $(@D)
    # The -MMD flags creates a .d file with the same name as the .o file.
    # The rdc enables relocatable device code (to have global vars defined in headers)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -MMD -c $< -o $@ 
    # -rdc=true it slows down code execution seriously!

.PHONY : clean	

run: $(BIN)
	$(BUILD_DIR)/$(BIN)

clean :
    # This should remove all generated files.
	-rm $(BUILD_DIR)/$(BIN) $(OBJ) $(DEP)

clobber: clean
