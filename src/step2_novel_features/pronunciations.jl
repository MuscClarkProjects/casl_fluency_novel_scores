using JSON
using Lazy
using PyCall

@pyimport nltk.corpus as nc

include("../helpers/helpers.jl")


getCmu() = nc.cmudict[:dict]()


typealias WordPronunciation{T<:AbstractString, A <: AbstractArray} Pair{T, A}
typealias WordPronunciations Vector{WordPronunciation}


masterPronunsFile() = getDataFile("step2", "pronunciations.json")

function createPronunciationsFile(
  extra_pronuns::WordPronunciations = WordPronunciations(),
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
    end
    [acc; words]
  end
end


findWordsWithoutPronunciations(pronuns_f = masterPronunsFile()
  ) = findWordsWithoutPronunciations(JSON.parsefile(pronuns_f))


findWordsWithoutPronunciations(pronuns::Dict=getCmu()) = getValidTaskWords(
  w -> !(w in keys(pronuns)))


function getMultiWordsPronunciations(multi_words=getValidTaskWords(w -> '_' in w),
  pronuns::Dict=getCmu())

  no_pronuns = ASCIIString[]
  ret = reduce(WordPronunciations(), multi_words) do acc::WordPronunciations, multi_word
    single_words = split(multi_word, '_')

    curr_no_pronuns = filter(w -> !(w in keys(pronuns)), single_words)
    if length(curr_no_pronuns) > 0
      for w in curr_no_pronuns
        push!(no_pronuns, w)
      end
      return acc
    end

    for w in single_words
      push!(acc, w=>pronuns[w])
    end

    acc
  end

  ret, no_pronuns
end
