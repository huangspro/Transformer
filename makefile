# 编译器和选项
CC = gcc
CFLAGS = -Wall -g -Iinclude -MMD

# 文件
SRC = $(wildcard src/*.C)
OBJ = $(SRC:.c=.o)
DEPS = $(OBJ:.o=.d)
TARGET = myprogram

# 默认目标
all: $(TARGET)

# 链接
$(TARGET): $(OBJ)
    $(CC) $(CFLAGS) -o $@ $^

# 编译规则
%.o: %.c
    $(CC) $(CFLAGS) -c $< -o $@

# 自动 include 依赖
-include $(DEPS)

# 清理
.PHONY: clean
clean:
    rm -f $(OBJ) $(DEPS) $(TARGET)