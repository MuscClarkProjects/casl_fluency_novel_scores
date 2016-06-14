project_folder = dirname(dirname(pwd()))


data_f(parts...) = joinpath(project_folder, "data", parts...)


is_valid_word(w::ASCIIString) = length(w) > 0 && !ismatch(r"^[@|?|!|\-|#|(]", w)


function load_words(f::AbstractString, delim::Union{AbstractString, Char}='\n')
  has_content::Bool = length(open(readall, f)) > 0
  if !has_content
    return ASCIIString[]
  end

  words::Vector{ASCIIString} = unique(readdlm(f, delim, ASCIIString))
  filter(is_valid_word, map(lowercase, words))
end


function load_words{T <: AbstractString}(fs::AbstractVector{T})
  typealias Words Vector{ASCIIString}

  all_words::Words = reduce(Words(), fs) do acc::Words, f::T
    words::Words = load_words(f)
    union(acc, words)
  end

  sort(all_words)
end


function get_all_dir_txt_files(dir)
  step1_dir = data_f("step1", dir)

  is_text_file(f::ASCIIString) = endswith(f, ".txt")
  add_path(f::AbstractString) = joinpath(step1_dir, f)

  ASCIIString[add_path(f) for f in filter(is_text_file, readdir(step1_dir))]
end


get_all_list_files() = get_all_dir_txt_files("lists")


get_all_meg_vf_files() = get_all_dir_txt_files("meg_vf")


typealias Strings Vector{ASCIIString}

function run{T <: AbstractString}(
  dest_file::Nullable{T}=Nullable(data_f("step1/targetwords/target_words.txt")))

  list_files::Strings = get_all_list_files()

  meg_files::Strings = get_all_meg_vf_files()

  all_word_files::Strings = [data_f("step1/targetwords/target_words_orig.txt"); list_files; meg_files]

  all_words::Strings = load_words(all_word_files)

  isnull(dest_file) || writedlm(get(dest_file), all_words)

  all_words
end


function find_non_matching_words(verbose::Bool=false, ignore_words...)
  ignore_words::Set{ASCIIString} = Set{ASCIIString}(ignore_words)

  orig_keys::Strings = begin
    f::ASCIIString = data_f("step1/pPMI_keys_old.txt")
    load_words(f, '\r')
  end

  word_files::Strings = union(get_all_meg_vf_files(), get_all_list_files())

  non_matchers = Strings()

  for f in word_files
    words::Strings = load_words(f)
    missing_words::Strings = filter(
      w -> !in(w, ignore_words), setdiff(words, orig_keys)
    )

    if verbose
      f_name = basename(f)
      length(missing_words > 0) || (println("file $f_name"))
      for w::ASCIIString in missing_words
        println("word: $w in $f_name")
        println("suggestions for $w")
        first_letters = length(w) > 4 ? w[1:4] : w[1:end-1]
        for k in sort(filter(i -> contains(i, first_letters), y))
          println("$k suggestion for $w")
        end
      end
    end

    non_matchers = union(non_matchers, missing_words)
  end

  non_matchers

end
