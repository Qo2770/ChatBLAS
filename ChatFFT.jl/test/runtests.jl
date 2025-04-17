import ChatFFT


const language = ChatFFT.ChatFFTPreferences.language
const model = ChatFFT.ChatFFTPreferences.model

if language == "c" && model == "openmp"
  println("--Running test for C and OpenMP")
  include("test_c_openmp.jl")
elseif language == "c" && model == "cuda"
  println("--Running test for C and CUDA")
  include("test_c_cuda.jl")
end
