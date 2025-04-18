echo -e 'using Pkg\nPkg.activate(".")\nPkg.instantiate()\nusing ChatFFT\nChatFFT.compilation()\ninclude("test/runtests.jl")\n' | julia

