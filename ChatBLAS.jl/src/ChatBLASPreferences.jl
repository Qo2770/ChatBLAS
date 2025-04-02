module ChatBLASPreferences

using Preferences

# taken from https://github.com/JuliaPackaging/Preferences.jl
function set_language(new_language::String)

    new_language_lc = lowercase(new_language)
    if !(new_language_lc in ("julia", "c", "fortran"))
        throw(ArgumentError("Invalid language: \"$(new_language)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("language" => new_language_lc)
    @info("New language set; restart your Julia session for this change to take effect!")
end

function set_model(new_model::String)

    new_model_lc = lowercase(new_model)
    if !(new_model_lc in ("openmp", "cuda", "hip"))
        throw(ArgumentError("Invalid model: \"$(new_model)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("model" => new_model_lc)
    @info("New model set; restart your Julia session for this change to take effect!")
end



const language = @load_preference("language", "c")
const model = @load_preference("model", "openmp")
const secret_key = @load_preference("secret_key", "")
#const gpt_model = @load_preference("gpt_model", "gpt-3.5-turbo")
#const gpt_model = @load_preference("gpt_model", "ft:gpt-3.5-turbo-1106:personal::8S4Je80c")
#const gpt_model = @load_preference("gpt_model", "ft:gpt-3.5-turbo-1106:personal:hip-blas1-chatblas:92hNz4Hj")
const gpt_model = @load_preference("gpt_model", "gpt-4o")

end # module ChatBLASPreferences
