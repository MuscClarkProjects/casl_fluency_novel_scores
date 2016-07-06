using JSON
using Lazy
using PyCall

@pyimport nltk.corpus as nc

include("../helpers/helpers.jl")
include("processPronunsHelpers.jl")


joinSylllables(syls::Matrix) = join(syls[1, :], '.')
joinSylllables(syls::Vector) = join(syls[1], '.')

function getCmu()
  cmu = nc.cmudict[:dict]()
  println("loaded cmu dict, will now take first row for each word")
  @>> cmu values map(joinSylllables) zip(keys(cmu)) Dict
end


typealias WordPronunciations{T<:AbstractString, A <: AbstractString} Dict{T, A}


function createPronunciationsFile(extra_pronuns::WordPronunciations;
    dest_f=masterPronunsFile())

  cmu = getCmu()
  for (word, pronuns) in extra_pronuns
    cmu[word] = pronuns
  end

  saveDict(cmu, dest_f)
end


function validTaskWords(extrafilter::Function = w -> true)
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


multiSplit(w) = split(w, r"-|_")
isMulti(w) = @> w multiSplit length Base.(:>)(1)

validSingleTaskWords() = validTaskWords(w -> !isMulti(w))
validMultiTaskWords() = validTaskWords(isMulti)


singleWordsWithoutPronunciations(pronuns_f = masterPronunsFile()
  ) = singleWordsWithoutPronunciations(JSON.parsefile(pronuns_f))


function singleWordsWithoutPronunciations(;pronuns::Dict=cmuWithPrevPronuns())
  @>> validSingleTaskWords() filter(w -> !(w in keys(pronuns)))
end


function multiWordsPronunciations(;multi_words=validMultiTaskWords(),
  pronuns::Dict=cmuWithPrevPronuns())

  no_pronuns = ASCIIString[]
  ret = reduce(Dict{ASCIIString, ASCIIString}(), multi_words) do acc::WordPronunciations, multi_word
    if multi_word in keys(pronuns)
      acc[multi_word] = pronuns[multi_word]
      return acc
    end

    single_words = multiSplit(multi_word)

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

prevNotInCmu() = @> "prev_not_in_cmu" readPronunsAsDict

newNotInCmu() = @> "new_not_in_cmu" readPronunsAsDict


cmuWithPrevPronuns(;cmu::Dict=getCmu(),
    prev_pronuns::Dict=prevNotInCmu()) = union(cmu, prev_pronuns) |> Dict


function taskWordsWithoutPronuns(;pronuns::Dict=cmuWithPrevPronuns())
  _, no_pronun_multi = multiWordsPronunciations(pronuns = pronuns)

  no_pronun_single = singleWordsWithoutPronunciations(pronuns = pronuns)

  union(no_pronun_multi, no_pronun_single) |> Set
end


taskWordsNotInCmuOrPrev(cmu::Dict = getCmu()) = taskWordsWithoutPronuns(
  cmuWithPrevPronuns(cmu=cmu)
)


function allKnownPronuns(;cmu::Dict=getCmu(),
  prev_not_in_cmu::Dict=prevNotInCmu(),
  new_not_in_cmu::Dict=newNotInCmu()
  )
  union(cmu, prev_not_in_cmu, new_not_in_cmu) |> Dict
end


function remainingTaskWords(cmu::Dict)
  all_pronuns = allKnownPronuns(cmu=cmu)
  taskWordsWithoutPronuns(pronuns=all_pronuns)
end


function calcMasterPronuns(;cmu::Dict=getCmu())
  all_pronuns = allKnownPronuns(cmu=cmu)

  function warnIfNotNone(arr, arr_name)
    num = arr |> length
    if num > 0
      warn("$num $(arr_name) without pronunciations")
    end
  end

  multi_words = begin
    ret, no_multi_pronuns = multiWordsPronunciations(pronuns=all_pronuns)

    warnIfNotNone(no_multi_pronuns, "multi-words")

    ret
  end

  single_words = begin
    single_no_pronuns = singleWordsWithoutPronunciations(pronuns=all_pronuns)

    warnIfNotNone(single_no_pronuns, "single words")

    [w=>all_pronuns[w] for w in validSingleTaskWords()]
  end

  union(single_words, multi_words) |> Dict
end


function writeMasterPronuns(f=pronunsCsv("master");cmu::Dict=getCmu())
  @>> calcMasterPronuns(cmu=cmu) writecsv(f)
end
