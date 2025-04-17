import LinearAlgebra
import ChatBLAS
using Test

@testset "Test Language" begin
    @test ChatBLAS.ChatBLASPreferences.language == "c"
end

@testset "Test Model" begin
    @test ChatBLAS.ChatBLASPreferences.model == "hip"
end

#@testset "Generating ChatBLAS codes" begin
#  @show pwd()
#  cd("../")
#  @show pwd()
#  ChatBLAS.BLAS1()
#  cd("test/")
#end

@testset "Building ChatBLAS library" begin
  @show pwd()
  cd("../")
  @show pwd()
  ChatBLAS.compilation()
  cd("test/")
  @show pwd()
end

@testset "Testing ChatBLAS routines" begin
end
@testset "SSCAL" begin
  n::Int32 = 100 
  a::Float32 = 5.0
  X = rand(Float32, n)
  X_ref = rand(Float32, n)
  X_ref .= X
  LinearAlgebra.BLAS.scal!(a, X_ref)
  @ccall "../ChatBLAS/libchatblas.so".chatblas_sscal(n::Cint, a::Cfloat, X::Ptr{Cfloat})::Cvoid
  @test isapprox(X, X_ref, rtol = 1e-6)
end
