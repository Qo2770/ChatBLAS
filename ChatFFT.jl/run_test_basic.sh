echo -e 'using Pkg\nPkg.activate(".")\nPkg.instantiate()\nusing ChatFFT\nChatFFT.FFT()\nChatFFT.compilation()\ninclude("test/runtests.jl")\n' | julia

