# ChatBLAS

Dependencies
---

- [julia](https://julialang.org/) = 1.9.1

Getting Started
---

Clone the repo and step into the `ChatBLAS.jl` directory:

```
git clone https://github.com/Qo2770/ChatBLAS.git
cd ChatBLAS/ChatBLAS.jl
```

Edit the `secret_key` inside `src/ChatBLASPreferences.jl` to be the OpenAI API Key:
```
const secret_key = @load_preference("secret_key", "$your_secret_key")
```

Running Correctness Tests
---

### On CPU

Edit the `model` inside `src/ChatBLASPreferences.jl` to be the following:
```
const model = @load_preference("model", "openmp")
```

Edit the `LocalPreferences.toml` file to target C and OpenMP:
```
[ChatBLAS]
language = "c"
model = "openmp"
```
Run the entire pipeline on CPU using OpenMP:
```
bash run_test_basic.sh
```

Compile and run tests on CPU using OpenMP:
```
bash run_test_only.sh
```

#### Expected Output
```
julia> include("test/runtests.jl")
--Running test for C and OpenMP
Test Summary: | Pass  Total  Time
Test Language |    1      1  0.0s
Test Summary: | Pass  Total  Time
Test Model    |    1      1  0.0s
Test Summary:             |Time
Testing ChatBLAS routines | None  0.0s
Test Summary: | Pass  Total  Time
SCOPY         |    1      1  0.3s
Test Summary: | Pass  Total  Time
SAXPY         |    1      1  0.0s
Test Summary: | Pass  Total  Time
SSCAL         |    1      1  0.0s
Test Summary: | Pass  Total  Time
SDOT          |    1      1  0.0s
Test Summary: | Pass  Total  Time
ISAMAX        |    1      1  0.0s
Test Summary: | Pass  Total  Time
SNRM2         |    1      1  0.0s
Test Summary: | Pass  Total  Time
SASUM         |    1      1  0.0s
Test Summary: | Pass  Total  Time
SSWAP         |    2      2  0.0s
Test Summary: | Pass  Total  Time
SDSDOT        |    1      1  0.0s
Test.DefaultTestSet("SDSDOT", Any[], 1, false, false, true, 1.744240484998517e9, 1.744240485018458e9, false)
```

### On GPU

Edit the `model` inside `src/ChatBLASPreferences.jl` to be the following:
```
const model = @load_preference("model", "cuda")
```

Edit the `LocalPreferences.toml` file to target CUDA on NVIDIA GPU:
```
[ChatBLAS]
language = "c"
model = "cuda"
```
Run the entire pipeline on NVIDIA GPU:
```
bash run_test_basic.sh
```

Compile and run tests on NVIDIA GPU:
```
bash run_test_only.sh
```


Running Performance Tests
---

### On CPU

*We recommend running performance tests on an Intel CPU as the baseline depends on Intel MKL library.* 

Copy the source files (*.c) to the `ChatBLAS/ChatBLAS.jl/ChatBLAS` directory and compile them with
```
make -f makefile.openmp
```
Navigate to `ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests`:
```
cd ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests
```
Run both the first command in `ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests/script` and the correct compilation line from that file. For example, on Intel CPU:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../
gcc saxpy_chatblas.c -m64 -I../inc/ -I/opt/intel/oneapi/mkl/latest/include -L../ -L/opt/intel/oneapi/mkl/latest/lib -lchatblas -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -o run
```
Run the following command:
```
./run
```

### On GPU

Copy the source files (*.cu) to the `ChatBLAS/ChatBLAS.jl/ChatBLAS` directory and compile them with
```
make -f makefile.cuda
```
Navigate to `ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests`:
```
cd ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests
```
Run both the first command in `ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests/script` and the correct compilation line from that file. For example, on NVIDIA GPU:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../
nvcc -lcublas -I../inc/ -L../ saxpy_chatblas.c -lchatblas -o run
```
Run the following command:
```
./run
```