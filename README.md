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

Running Correctness Tests on CPU
---

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

### Expected Output
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

Running Performance Tests on CPU
---

We recommend running performance tests on an Intel CPU as the baseline depends on Intel MKL library. Follow these steps to run the performance tests:
1. Copy the source files (*.c or *.cu) to the `ChatBLAS/ChatBLAS.jl/ChatBLAS` directory
2. Compile them using the makefile
3. Navigate to `ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests`
4. Run both the first command in `ChatBLAS/ChatBLAS.jl/ChatBLAS/performance-tests/script` and the correct compilation line from that file
5. Then run the following command:
```
./run
```