using JSON
using Lazy
using StatsBase

include("../helpers/helpers.jl")


immutable WordLocation
  file::ASCIIString
  index::Int64
end


typealias WordLocations Dict{ASCIIString, Vector{WordLocation}}
typealias TaskWordLocations Dict{ASCIIString, WordLocations}

function calculateTaskWordLocations()
  taskwords = TaskWordLocations()

  for task_f in allTaskFiles()
    task = getTask(task_f)
    taskwords[task] = wordlocations = get(taskwords, task, WordLocations())

    if filesize(task_f) == 0
      continue
    end

    valid_words = begin
      all_words = readcsv(task_f, ASCIIString, comments=false)[:]
      filter(isValidWord, all_words)
    end

    for (rank, word) in enumerate(valid_words)
      if isComment(word)
        continue
      end

      wordlocations[word] = get(wordlocations, word, WordLocation[])
      push!(wordlocations[word], WordLocation(basename(task_f), rank))
    end
  end

  taskwords
end


saveTaskWordLocations(twl::TaskWordLocations,
  dest_f=getDataFile("step2", "word_locations.json")) = saveDict(twl, dest_f)


typealias MeanStd Tuple{Float64, Float64}

typealias WordMeanStds Dict{ASCIIString, MeanStd}
typealias TaskWordMeanStds Dict{ASCIIString, WordMeanStds}

function convertTaskWordLocationsToMeanStd(twl::TaskWordLocations)
  ret = TaskWordMeanStds()

  for (task, wordlocations) in twl
    ret[task] = get(ret, task, WordMeanStds())

    for (word, wordlocations) in wordlocations
      ret[task][word] = map(wl -> wl.index, wordlocations) |>  mean_and_std
    end
  end

  ret
end


function saveTaskWordMeanStd(twl::TaskWordLocations,
    dest_f=getDataFile("step2", "word_locations_mean_std.json"))
  saveTaskWordMeanStd(convertTaskWordLocationsToMeanStd(twl), dest_f)
end


saveTaskWordMeanStd(twm::TaskWordMeanStds,
    dest_f=getDataFile("step2", "word_locations_mean_std.json")) = saveDict(
  twm, dest_f)
