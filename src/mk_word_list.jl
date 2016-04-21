data_f(f_name::AbstractString) = joinpath(dirname(pwd()), "data", f_name)


is_valid_word(w::ASCIIString) = length(w) > 0 && !ismatch(r"^[!|\-|#|(]", w)


function load_words(f::AbstractString)
  has_content::Bool = length(open(readall, f)) > 0
  if !has_content
    return ASCIIString[]
  end

  words::Vector{ASCIIString} = unique(readdlm(f, ASCIIString))
  filter(is_valid_word, words)
end


function load_words{T <: AbstractString}(fs::AbstractVector{T})
  typealias Words Vector{ASCIIString}

  all_words::Words = reduce(Words(), fs) do acc::Words, f::T
    words::Words = load_words(f)
    union(acc, words)
  end

  sort(all_words)
end

function run{T <: AbstractString}(
    dest_file::Nullable{T}=Nullable(data_f("step1/target_words.txt")))

  typealias Strings Vector{ASCIIString}

  list_files::Strings = begin
    step1_lists_dir = data_f("step1/lists/")

    is_text_file(f::ASCIIString) = endswith(f, ".txt")
    add_path(f::AbstractString) = joinpath(step1_lists_dir, f)

    ASCIIString[add_path(f) for f in filter(is_text_file, readdir(step1_lists_dir))]
  end

  all_word_files::Strings = [data_f("step1/target_words_orig.txt"); list_files]

  all_words::Strings = load_words(all_word_files)

  isnull(dest_file) || writedlm(get(dest_file), all_words)

  all_words
end
