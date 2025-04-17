import LinearAlgebra
import ChatFFT
using FFTW
using Test

testdir = @__DIR__
const libpath = normpath(joinpath(testdir, "..", "ChatFFT", "libfft.so"))

@testset "Test Language" begin
    @test ChatFFT.ChatFFTPreferences.language == "c"
end

@testset "Test Model" begin
    @test ChatFFT.ChatFFTPreferences.model == "openmp"
end

#@testset "Generating ChatBLAS codes" begin
#  @show pwd()
#  cd("../")
#  @show pwd()
#  ChatBLAS.BLAS1()
#  cd("test/")
#end

#@testset "Building ChatBLAS library" begin
#  @show pwd()
#  cd("../")
#  @show pwd()
#  ChatBLAS.compilation()
#  cd("test/")
#  @show pwd()
#end

@testset "Testing FFT routines" begin

  @testset "Inverse" begin
    n::Int32 = 8 
    isign::Int32 = -1
    isign_rev::Int32 = 1
    X = zeros(16) 
    X[begin:2:end] = rand(Float32, n)
    ref = X
    @ccall "$libpath".chatfft(X::Ptr{Cfloat}, n::Cint, isign::Cint)::Cvoid
    @ccall "$libpath".chatfft(X::Ptr{Cfloat}, n::Cint, isign_rev::Cint)::Cvoid
    @test isapprox(X, ref, rtol = 1e-6)
  end
  
  @testset "Rand FFTW" begin
    n::Int32 = 4
    isign::Int32 = -1
    X = Float32[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
    ref = fft(X[begin:2:end])  
    @ccall "$libpath".chatfft(X::Ptr{Cfloat}, n::Cint, isign::Cint)::Cvoid
    @test isapprox(X[begin:2:end], real.(ref), rtol = 1e-6)
    @test isapprox(X[2:2:end], imag.(ref), rtol = 1e-6)
  end

end
