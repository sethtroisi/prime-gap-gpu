# Copyright 2025-2026 Seth Troisi
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
OBJS	= gap_common.o gap_search_common.o \
	  overflow.o xoroshiro128plus.o \
	  gpu_testing.o
PRIMORIAL_OBJS = gap_primorial_testing.o gpu_primorial_sieve.o
# TODO gpu_primorial_sieve.o
LINEAR_OBJS = gap_linear_testing.o
OUT	= gap_search_linear
CC	= g++
CFLAGS	= $(OPT) -Wall -Werror -Wno-vla -mtune=native -flto
NVCC	= nvcc
ARCH    = sm_89
CUDA_FLAGS	= $(OPT) -arch=$(ARCH) --resource-usage \
		  -Xcompiler -Wall,-Werror,-mtune=native

BITS    = 1024
MP      = 66

LDFLAGS	= -lgmp -lprimesieve -lcudart -flto=auto
# Need for local gmp / primesieve
#LDFLAGS+= -L /usr/local/lib

all: $(OUT)

gpu_primorial_sieve.o: gpu_primorial_sieve.cu
	$(NVCC) $^ -o $@ -DMP=$(MP) -c $(CUDA_FLAGS)

gpu_testing.o: gpu_testing.cu
	$(NVCC) $^ -o $@ -c -DGPU_BITS=$(BITS) -DMP=$(MP) $(CUDA_FLAGS) -I../CGBN/include

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS) $(DEFINES)


gap_search_primorial: gap_search_primorial.cpp $(OBJS) $(PRIMORIAL_OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

gap_search_linear: gap_search_linear.cpp $(OBJS) $(LINEAR_OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

gpu_benchmark: gpu_benchmark.cu gap_common.o
	$(NVCC) $^ -o $@ -DGPU_BITS=$(BITS) -DMP=$(MP) $(CUDA_FLAGS) -I../CGBN/include -lgmp

.PHONY: all clean

clean:
	rm -f $(OUT) *.o gpu_benchmark
