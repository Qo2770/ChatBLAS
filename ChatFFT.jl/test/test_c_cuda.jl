import LinearAlgebra
import ChatFFT
using Test

testdir = @__DIR__
const libpath = normpath(joinpath(testdir, "..", "ChatFFT", "libfft.so"))

@testset "Test Language" begin
    @test ChatFFT.ChatFFTPreferences.language == "c"
end

@testset "Test Model" begin
    @test ChatFFT.ChatFFTPreferences.model == "cuda"
end

#@testset "Generating ChatBLAS codes" begin
#  @show pwd()
#  cd("../")
#  @show pwd()
#  ChatBLAS.BLAS1()
#  cd("test/")
#end

# @testset "Building ChatBLAS library" begin
#   @show pwd()
#   cd("../")
#   @show pwd()
#   ChatBLAS.compilation()
#   cd("test/")
#   @show pwd()
# end

# Nesting all function tests inside one larger test set ensures that
# if one test set fails, subsequent test sets will still attempt to run.
@testset "FFT Function Tests" begin
  @testset "Testing FFT routines" begin
  end

  @testset "Inverse" begin
    n::Int32 = 64 
    isign::Int32 = -1
    isign_rev::Int32 = 1
    X = rand(Float32, n)
    ref = X
    @ccall "$libpath".chatfft(X::Ptr{Cfloat}, n::Cint, isign::Cint)::Cvoid
    @ccall "$libpath".chatfft(X::Ptr{Cfloat}, n::Cint, isign_rev::Cint)::Cvoid
    @test isapprox(X, ref, rtol = 1e-6)
  end

end
