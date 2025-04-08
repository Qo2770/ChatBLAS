module ChatBLAS

using OpenAI

# module to set back end preferences
include("ChatBLASPreferences.jl")

function my_create_chat(api_key, model_id, messages)
  provider = OpenAI.OpenAIProvider(api_key, "https://cmu.litellm.ai", "")
  return OpenAI.create_chat(provider, model_id, messages)
end

#BLAS level 1
function scopy()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function that copies a vector x to a vector y. Vectors are length n, use C and OpenMP to compute in parallel, use next function name and parameters void chatblas_scopy(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function that copies a vector x to a vector y. Vectors are length n, use C and CUDA to compute in parallel, use next function name and parameters void chatblas_scopy(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt = "give me a kernel and a fuction that calls the kernel only, no comments, no main, that copies a vector x to a vector y, vectors are length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void scopy_kernel( int n, float *x, float *y ) { and the next function name and parameters for the function void chatblas_scopy(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/scopy.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/scopy.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/scopy.cpp", "w") do file write(file, string) end
  end

end

function sswap()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function code only that computes the swap of two vectors x and y. Vectors are length n, use C and OpenMP to compute in parallel, use next function name and parameters void chatblas_sswap(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function code only that computes the swap of two vectors x and y. Vectors are length n, use C and CUDA to compute in parallel, use next function name and parameters void chatblas_sswap(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt = "give me a kernel and a fuction that calls the kernel only, no comments, no main, that computes the swap of two vectors x and y, vectors are length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void sswap_kernel(int n, float *x, float *y) { and the next function name and parameters for the function void chatblas_sswap(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/sswap.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/sswap.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/sswap.cpp", "w") do file write(file, string) end
  end

end

function sdot()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function code only that computes and returns the dot product of two vectors x and y. Vectors are length n, use C and OpenMP to compute in parallel, use next function name and parameters float chatblas_sdot(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function code only that computes and returns the dot product of two vectors x and y. Vectors are length n, use C and CUDA to compute in parallel, use next function name and parameters float chatblas_sdot(int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt = "give me a kernel and a fuction that calls the kernel only, no comments, no main, that computes and return the dot product of two vectors x and y, vectors are length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void sdot_kernel(int n, float *x, float *y, float *res) { where the result of computing the dot product of the vector x and y is returned in the pointer res and the next function name and parameters for the function float chatblas_sdot( int n, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/sdot.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/sdot.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/sdot.cpp", "w") do file write(file, string) end
  end

end

function sdsdot()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function code that computes and returns the dot product of two vectors x and y. The final result must add the scalar b which is passed as argument. The accomulation is computed in double precision, so the elements of the vector X and Y must be casted to float before the computation. Vectors are length n, use C and OpenMP to compute in parallel, use next function name and parameters float chatblas_sdsdot(int n, float b, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function code that computes and returns the dot product of two vectors x and y. The final result must add the scalar b which is passed as argument. The accomulation is computed in double precision, so the elements of the vector X and Y must be casted to float before the computation. Vectors are length n, use C and CUDA to compute in parallel, use next function name and parameters float chatblas_sdsdot(int n, float b, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt = "give me a kernel and a fuction that calls the kernel only, no comments, no main, that computes and return the dot product of two vectors x and y plus a scalar b. The accomulation is computed in double precision. Vectors are length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void sdsdot_kernel( int n, float b, float *x, float *y, float *res ) { where the result of computing the dot product of the vector x and y plus b is returned in the pointer res and the next function name and parameters for the function float chatblas_sdsdot( int n, float b, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/sdsdot.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/sdsdot.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/sdsdot.cpp", "w") do file write(file, string) end
  end

end

function snrm2()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt ="Give me a function code only that computes and returns the Euclidean norm of a vector x. Vectors are length n, use C and OpenMP to compute in parallel, use next function name and parameters float chatblas_snrm2(int n, float *x) {. Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt ="Give me a function code only that computes and returns the Euclidean norm of a vector x. Vectors are length n, use C and CUDA to compute in parallel, use next function name and parameters float chatblas_snrm2(int n, float *x) {. Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt = "give me a kernel and a fuction that calls the kernel only, no comments, no main, that computes and return the Euclidean norm of a vector x. Vector is length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void snrm2_kernel( int n, float *x, float *res) { where the result of computing the Ecludian norm of the vector x is returned in the pointer res, and the next function name and parameters for the function float chatblas_snrm2(int n, float *x) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/snrm2.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/snrm2.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/snrm2.cpp", "w") do file write(file, string) end
  end

end

function sscal()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function code that scales a vector x by a constant a, vector x is length n, use C and OpenMP to compute in parallel, use next function name and parameters void chatblas_sscal( int n, float a , float *x) {. Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function code that scales a vector x by a constant a, vector x is length n, use C and CUDA to compute in parallel, use next function name and parameters void chatblas_sscal( int n, float a , float *x) {. Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt = "give me a kernel and a fuction that calls the kernel only, no comments, no main, that scales a vector x by a constant a, vector x is length n, use C and HIP to compute in parallel, allocate and free the GPU vector and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void sscal_kernel( int n, float a , float *x ) { and the next function name and parameters for the function void chatblas_sscal( int n, float a, float *x) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/sscal.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/sscal.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/sscal.cpp", "w") do file write(file, string) end
  end

end

function saxpy()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function code only that computes a multiplication of a vector x by a constant a and the result is added to a vector y. Vectors x and y are length n, use C and OpenMP to compute in parallel include the next line in the code, use next function name and parameters void chatblas_saxpy(int n, float a, float *x, float *y). Inlcude the next line at the beginning #include \"chatblas_openmp.h\""		
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function code only that computes a multiplication of a vector x by a constant a and the result is added to a vector y. Vectors x and y are length n, use C and CUDA to compute in parallel include the next line in the code, use next function name and parameters void chatblas_saxpy(int n, float a, float *x, float *y). Inlcude the next line at the beginning #include \"chatblas_cuda.h\""		
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt =  "Only respond with code as plain text without code block syntax around it. give me a kernel and a function only, no comments, no main, that computes a multiplication of a vector x by a constant a and the result is added to a vector y. Vectors x and y are length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void saxpy_kernel(int n, float a, float *x, float *y) { and the next function name and parameters for the function void chatblas_saxpy(int n, float a, float *x, float *y) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""		
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/saxpy.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/saxpy.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/saxpy.cpp", "w") do file write(file, string) end
  end

end

function isamax()
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function code only that returns the position of the element of a vector X that has the largest absolute value. Vector X is length n, use C and OpenMP to compute in parallel include the next line in the code, use next function name and parameters int chatblas_isamax(int n, float *x). Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function code only that returns the position of the element of a vector X that has the largest absolute value. Vector X is length n, use C and CUDA to compute in parallel include the next line in the code, use next function name and parameters int chatblas_isamax(int n, float *x). Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt =  "give me a kernel and a function only, no comments, no main, that returns the position of the element of a vector X that has the largest absolute value. Vectors x is length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void isamax_kernel(int n, float *x, float *ind) { where the position of the largest absolute value of vector x is returned using the ind pointer and the next function name and parameters for the function int chatblas_isamax(int n, float *x) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""		
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/isamax.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/isamax.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/isamax.cpp", "w") do file write(file, string) end
  end

end

function sasum()

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    prompt = "Give me a function code only that returns the sum of the absolute values of the elements of a vector x. Vector x is length n, use C and OpenMP to compute in parallel include the next line in the code, use next function name and parameters float chatblas_sasum(int n, float *x). Inlcude the next line at the beginning #include \"chatblas_openmp.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    prompt = "Give me a function code only that returns the sum of the absolute values of the elements of a vector x. Vector x is length n, use C and CUDA to compute in parallel include the next line in the code, use next function name and parameters float chatblas_sasum(int n, float *x). Inlcude the next line at the beginning #include \"chatblas_cuda.h\""
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    prompt =  "give me a kernel and a function only, no comments, no main, that returns the sum of the absolute values of the elements of a vector x. Vectors x is length n, use C and HIP to compute in parallel, allocate and free the GPU vectors and make the CPU - GPU memory transfers in the function, use next function name and parameters for the kernel __global__ void sasum_kernel(int n, float *x, float *sum) { where the sum of the absolute values of the elements of the vector x is returned using the sum pointer and the next function name and parameters for the function float chatblas_sasum(int n, float *x) {. Inlcude the next line at the beginning #include \"chatblas_hip.h\""		
  end

  r = my_create_chat(ChatBLASPreferences.secret_key, ChatBLASPreferences.gpt_model,[Dict("role" => "user", "content"=> prompt)])

  string = r.response[:choices][begin][:message][:content]
  string = replace(string, r"^(.*\n)*?(?=(#include))" => "")
  string = replace(string, r"[^}]+$" => "")
  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp" 
    open("ChatBLAS/sasum.c", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    open("ChatBLAS/sasum.cu", "w") do file write(file, string) end
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    open("ChatBLAS/sasum.cpp", "w") do file write(file, string) end
  end

end

function BLAS1()
  sscal()
  saxpy()
  scopy()
  sdot()
  sdsdot()
  sswap()
  sasum()
  isamax()
  snrm2()
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

  if ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "openmp"
    cd("ChatBLAS")
    run(`make clean -f makefile.openmp`) 
    run(`make -f makefile.openmp`) 
    cd("..")
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "cuda" 
    cd("ChatBLAS")
    run(`make clean -f makefile.cuda`) 
    run(`make -f makefile.cuda`) 
    cd("..") 
  elseif ChatBLASPreferences.language == "c" && ChatBLASPreferences.model == "hip" 
    cd("ChatBLAS")
    run(`make clean -f makefile.hip`) 
    run(`make -f makefile.hip`) 
    cd("..") 
  end

end

function __init__()
  @info("Using ChatBLAS language: $(ChatBLASPreferences.language)")
  @info("Using ChatBLAS model: $(ChatBLASPreferences.model)")

end

end # module ChatBLAS
