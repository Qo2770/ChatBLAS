import ChatBLAS


const language = ChatBLAS.ChatBLASPreferences.language
const model = ChatBLAS.ChatBLASPreferences.model

if language == "c" && model == "openmp"
  println("--Running test for C and OpenMP")
  include("test_c_openmp.jl")
elseif language == "c" && model == "cuda"
  println("--Running test for C and CUDA")
  include("test_c_cuda.jl")
elseif language == "c" && model == "hip"
  println("--Running test for C and HIP")
  include("test_c_hip.jl")
end
