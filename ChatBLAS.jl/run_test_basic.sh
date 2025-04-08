echo -e 'using Pkg\nPkg.activate(".")\nPkg.instantiate()\nusing ChatBLAS\nChatBLAS.BLAS1()\nChatBLAS.compilation()\ninclude("test/runtests.jl")\n' | julia

