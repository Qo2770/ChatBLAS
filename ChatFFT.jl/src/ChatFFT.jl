module ChatFFT

using OpenAI

# module to set back end preferences
include("ChatFFTPreferences.jl")

function my_create_chat(api_key, model_id, messages)
  provider = OpenAI.OpenAIProvider(api_key, "https://cmu.litellm.ai", "")
  return OpenAI.create_chat(provider, model_id, messages)
end

function fft()

  if ChatFFTPreferences.language == "c" && ChatFFTPreferences.model == "openmp" 
    prompt = "Give me a function that applies a complex-to-complex Fast Fourier Transform on a vector x in-place. The vector is of length n, and a parameter isign should be set such that -1 constitutes the forward transform and 1 the inverse FFT. Use C and OpenMP but no FFT library like FFTW to compute in parallel, use next function name and parameters void chatfft(float *x, int n, int isigm) {. Include the next line at the beginning #include \"chatfft_openmp.h\""
  elseif ChatFFTPreferences.language == "c" && ChatFFTPreferences.model == "cuda" 
    prompt = ""
  end

  r = my_create_chat(ChatFFTPreferences.secret_key, ChatFFTPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(```c)" => "")
  string = replace(string, r"(```\n)(.*\n)*.*$" => "")
  if ChatFFTPreferences.language == "c" && ChatFFTPreferences.model == "openmp" 
    open("ChatFFT/fft.c", "w") do file write(file, string) end
  elseif ChatFFTPreferences.language == "c" && ChatFFTPreferences.model == "cuda" 
    open("ChatFFT/fft.cu", "w") do file write(file, string) end
  end

end

function FFT()
  fft()
end

#function makefile()

#  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
#    prompt =  "give me a Makefile file only, do not add comments, that compiles the next files: saxpy.c, sscal.c, using gcc compiler using -c, openmp and O3 flags"		
#  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
#    prompt =  "give me a Makefile that compiles the next files: saxpy.cu, sscal.cu with nvcc compiler using cuda and O3 flags"		
#  end

#  r = create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

#  string = r.response[:choices][begin][:message][:content]
#  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
#    open("ChatBLAS/Makefile", "w") do file write(file, string) end
#  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
#    open("ChatBLAS/Makefile", "w") do file write(file, string) end
#  end

#end

function compilation()

  if ChatFFTPreferences.language == "c" && ChatFFTPreferences.model == "openmp"
    cd("ChatFFT")
    run(`make clean -f makefile.openmp`) 
    run(`make -f makefile.openmp`) 
    cd("..")
  elseif ChatFFTPreferences.language == "c" && ChatFFTPreferences.model == "cuda" 
    cd("ChatFFT")
    run(`make clean -f makefile.cuda`) 
    run(`make -f makefile.cuda`) 
    cd("..") 
  end

end

function __init__()
  @info("Using FFT language: $(ChatFFTPreferences.language)")
  @info("Using FFT model: $(ChatFFTPreferences.model)")

end

end # module ChatBLAS
