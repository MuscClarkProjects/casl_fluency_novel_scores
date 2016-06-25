using JSON
using Lazy
using PyCall

@pyimport nltk.corpus as nc

include("../helpers/helpers.jl")


getCmu() = nc.cmudict[:dict]()


typealias WordPronunciations{T<:AbstractString, A <: AbstractArray} Dict{T, A}


masterPronunsFile() = getDataFile("step2", "pronunciations.json")

function createPronunciationsFile(extra_pronuns::WordPronunciations;
    dest_f=masterPronunsFile())

  cmu = getCmu()
  for (word, pronuns) in extra_pronuns
    cmu[word] = pronuns
  end

  saveDict(cmu, dest_f)
end


function getValidTaskWords(extrafilter::Function = w -> true)
  fs::Vector{ASCIIString} = filter(f -> filesize(f) > 0, allTaskFiles())

  reduce(ASCIIString[], fs) do acc::Vector{ASCIIString}, f::ASCIIString
    words = @>> readcsv(f, ASCIIString)[:] begin
      filter(isValidWord)
      filter(extrafilter)
      map(lowercase)
    end
    [acc; words]
  end
end


findSimpleWordsWithoutPronunciations(pronuns_f = masterPronunsFile()
  ) = findSimpleWordsWithoutPronunciations(JSON.parsefile(pronuns_f))


findSimpleWordsWithoutPronunciations(;pronuns::Dict=combineCmuWithPrevPronuns()) = getValidTaskWords(
  w -> !('_' in w) && !(w in keys(pronuns)))


function getMultiWordsPronunciations(;multi_words=getValidTaskWords(w -> '_' in w),
  pronuns::Dict=combineCmuWithPrevPronuns())

  no_pronuns = ASCIIString[]
  ret = reduce(Dict{ASCIIString, Array}(), multi_words) do acc::WordPronunciations, multi_word
    if multi_word in keys(pronuns)
      acc[multi_word] = pronuns[multi_word]
      return acc
    end

    single_words = split(multi_word, r"_|-")

    curr_no_pronuns = filter(w -> !(w in keys(pronuns)), single_words)
    if length(curr_no_pronuns) > 0
      for w in curr_no_pronuns
        push!(no_pronuns, w)
      end
      return acc
    end

    for w in single_words
      acc[w] = pronuns[w]
    end

    acc
  end

  ret, no_pronuns
end


function getPrevPronuns()
  pro_dir = getDataFile("step1", "pronunciations")
  fs = readdir(pro_dir)

  reduce(Dict{ASCIIString, Array}(), fs) do words::WordPronunciations, f::AbstractString
    f_pronuns = readcsv(joinpath(pro_dir, f), ASCIIString)
    num_words, _ = size(f_pronuns)
    for i in 1:num_words
      words[f_pronuns[i, 1]] = [f_pronuns[i, 2]]
    end
    words
  end
end


combineCmuWithPrevPronuns(;cmu::Dict=getCmu(),
    prev_pronuns::Dict=getPrevPronuns()) = union(cmu, prev_pronuns) |> Dict


function getAllTaskWordsWithoutPronuns(;
  pronuns::Dict=combineCmuWithPrevPronuns())

  _, no_pronun_multi = getMultiWordsPronunciations(pronuns = pronuns)

  no_pronun_simple = findSimpleWordsWithoutPronunciations(pronuns = pronuns)

  union(no_pronun_multi, no_pronun_simple) |> Set

end
