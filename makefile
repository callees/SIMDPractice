COMPILER := g++
COMPILER_FLAG := -g  -std=c++20 -mavx2
LIB_FLAGS :=

BUILD_DIR := build
SRC_DIR := src

ALL_SRCS := $(wildcard src/*.cpp)
SRCS := $(patsubst src/%, %, $(ALL_SRCS))
OBJS := $(foreach src, $(SRCS), $(BUILD_DIR)/$(subst .cpp,.o,$(src)))

MAIN_PROG := $(BUILD_DIR)/simdprog

.PHONY: all
all: $(MAIN_PROG)

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)/*

.PHONY: what
what:
	@echo "Source Files: $(SRCS)"
	@echo "Object Files: $(OBJS)"
	@echo "Main Program: $(MAIN_PROG)"
	@echo "Library Directory: $(LIB_DIR)"
	@echo "Libraries: $(LIBS)"

$(MAIN_PROG): $(OBJS)
	@echo "Building main program"
	$(COMPILER) $(COMPILER_FLAG) $(OBJS) $(LIB_FLAGS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@echo "Building objects"
	$(COMPILER) $(COMPILER_FLAG) -c $< -o $@