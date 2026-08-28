# Copyright 2025 Seth Troisi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OPT     = -O3 -std=c++20 -g
OBJS	= gap_common.o gpu_testing.o overflow.o
OUT	= gap_search_gpu
CC	= g++
CFLAGS	= $(OPT) -Wall -Werror -Wno-vla -mtune=native
NVCC	= nvcc
ARCH    = sm_89
CUDA_FLAGS	= $(OPT) -arch=$(ARCH) --resource-usage \
		  -Xcompiler -Wall \
	          -Xcompiler -Werror

BITS    = 1024

LDFLAGS	= -lgmp -lprimesieve -lcudart
# Need for local gmp / primesieve
#LDFLAGS+= -L /usr/local/lib

all: $(OUT)

sieve_small.o: sieve_small.cu
	nvcc $^ -o $@ -c $(CUDA_FLAGS)

gpu_testing.o: gpu_testing.cu
	nvcc $^ -o $@ -c -DGPU_BITS=$(BITS) $(CUDA_FLAGS) -I../CGBN/include

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS) $(DEFINES)


gap_search_gpu: gap_search_gpu.cpp $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

gpu_benchmark: gpu_benchmark.cu gap_common.o
	nvcc $^ -o $@ -DGPU_BITS=$(BITS) $(CUDA_FLAGS) -I../CGBN/include $(LDFLAGS)

.PHONY: all clean

clean:
	rm -f $(OUT) *.o gpu_benchmark
