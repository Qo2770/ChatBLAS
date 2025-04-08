echo -e 'using Pkg\nPkg.activate(".")\nPkg.instantiate()\nusing ChatBLAS\nChatBLAS.compilation()\ninclude("test/runtests.jl")\n' | julia

