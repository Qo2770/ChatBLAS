import LinearAlgebra
import ChatBLAS
using Test

testdir = @__DIR__
const libpath = normpath(joinpath(testdir, "..", "ChatBLAS", "libchatblas.so"))

@testset "Test Language" begin
    @test ChatBLAS.ChatBLASPreferences.language == "c"
end

@testset "Test Model" begin
    @test ChatBLAS.ChatBLASPreferences.model == "openmp"
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

@testset "Testing ChatBLAS routines" begin
end

@testset "SCOPY" begin
  n::Int32 = 100 
  X = rand(Float32, n)
  Y = rand(Float32, n)
  X_ref = rand(Float32, n)
  Y_ref = rand(Float32, n)
  X_ref .= X
  Y_ref .= Y
  LinearAlgebra.BLAS.blascopy!(n, X_ref, 1, Y_ref, 1)
  @ccall "$libpath".chatblas_scopy(n::Cint, X::Ptr{Cfloat}, Y::Ptr{Cfloat})::Cvoid
  @test isapprox(Y, Y_ref, rtol = 1e-6)
end

@testset "SAXPY" begin
  n::Int32 = 100 
  a::Float32 = 5.0
  X = rand(Float32, n)
  Y = rand(Float32, n)
  X_ref = rand(Float32, n)
  Y_ref = rand(Float32, n)
  X_ref .= X
  Y_ref .= Y
  LinearAlgebra.BLAS.axpy!(a, X_ref, Y_ref)
  @ccall "$libpath".chatblas_saxpy(n::Cint, a::Cfloat, X::Ptr{Cfloat}, Y::Ptr{Cfloat})::Cvoid
  @test isapprox(Y, Y_ref, rtol = 1e-6)
end

@testset "SSCAL" begin
  n::Int32 = 100 
  a::Float32 = 5.0
  X = rand(Float32, n)
  X_ref = rand(Float32, n)
  X_ref .= X
  LinearAlgebra.BLAS.scal!(a, X_ref)
  @ccall "$libpath".chatblas_sscal(n::Cint, a::Cfloat, X::Ptr{Cfloat})::Cvoid
  @test isapprox(X, X_ref, rtol = 1e-6)
end

@testset "SDOT" begin
  n::Int32 = 100 
  X = rand(Float32, n)
  Y = rand(Float32, n)
  X_ref = rand(Float32, n)
  Y_ref = rand(Float32, n)
  X_ref .= X
  Y_ref .= Y
  res::Float32 = 0.0
  res_ref::Float32 = res
  res_ref = LinearAlgebra.dot(X_ref, Y_ref)
  res = @ccall "$libpath".chatblas_sdot(n::Cint, X::Ptr{Cfloat}, Y::Ptr{Cfloat})::Cfloat
  @test isapprox(res, res_ref, rtol = 1e-6)
end

@testset "ISAMAX" begin
  n::Int32 = 100 
  X = rand(Float32, n)
  X_ref = rand(Float32, n)
  X_ref .= X
  ret::Int32 = 0.0
  ret_ref::Int32 = ret
  ret_ref = LinearAlgebra.BLAS.iamax(X_ref)
  ret = @ccall "$libpath".chatblas_isamax(n::Cint, X::Ptr{Cfloat})::Cint
  @test isapprox(ret, ret_ref, atol = 1.0)
end

@testset "SNRM2" begin
  n::Int32 = 100 
  X = rand(Float32, n)
  X_ref = rand(Float32, n)
  X_ref .= X
  ret::Float32 = 0.0
  ret_ref::Float32 = ret
  ret_ref = LinearAlgebra.BLAS.nrm2(n, X_ref, 1)
  ret = @ccall "$libpath".chatblas_snrm2(n::Cint, X::Ptr{Cfloat})::Cfloat
  @test isapprox(ret, ret_ref, rtol = 1e-6)
end

@testset "SASUM" begin
  n::Int32 = 100 
  X = rand(Float32, n)
  X_ref = rand(Float32, n)
  X_ref .= X
  ret::Float32 = 0.0
  ret_ref::Float32 = ret
  ret_ref = LinearAlgebra.BLAS.asum(n, X_ref, 1)
  ret = @ccall "$libpath".chatblas_sasum(n::Cint, X::Ptr{Cfloat})::Cfloat
  @test isapprox(ret, ret_ref, rtol = 1e-6)
end

@testset "SSWAP" begin
  function sswap(n::Int32, X::Vector{Float32}, Y::Vector{Float32})
    tmp::Float32 = 0.0
    for i in 1:n
      tmp = Y[i]
      Y[i] = X[i]
      X[i] = tmp
    end 
  end
  n::Int32 = 100 
  X = rand(Float32, n)
  Y = rand(Float32, n)
  X_ref = rand(Float32, n)
  Y_ref = rand(Float32, n)
  X_ref .= X
  Y_ref .= Y
  sswap(n, X_ref, Y_ref)
  @ccall "$libpath".chatblas_sswap(n::Cint, X::Ptr{Cfloat}, Y::Ptr{Cfloat})::Cvoid
  @test isapprox(X, X_ref, atol = 1.0)
  @test isapprox(Y, Y_ref, atol = 1.0)
end

@testset "SDSDOT" begin
  function sdsdot(n::Int32, b::Float32, X::Vector{Float32}, Y::Vector{Float32})
    tmp::Float32 = 0.0
    X64 = convert(Vector{Float64}, X)
    Y64 = convert(Vector{Float64}, Y)
    for i in 1:n
      tmp = tmp + X64[i] * Y64[i]
    end 
    return tmp + b
  end
  n::Int32 = 100 
  b::Float32 = 3.0 
  X = rand(Float32, n)
  Y = rand(Float32, n)
  X_ref = rand(Float32, n)
  Y_ref = rand(Float32, n)
  X_ref .= X
  Y_ref .= Y
  ret_ref = sdsdot(n, b, X_ref, Y_ref)
  ret = @ccall "$libpath".chatblas_sdsdot(n::Cint, b::Cfloat, X::Ptr{Cfloat}, Y::Ptr{Cfloat})::Cfloat
  @test isapprox(ret, ret_ref, rtol = 1e-6)
end
