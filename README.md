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

Running on CPU
---

Edit the `LocalPreferences.toml` file to target C and OpenMP:
```
[ChatBLAS]
language = "c"
model = "openmp"
```
Run tests on CPU using OpenMP:
```
bash run_test_basic.sh
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