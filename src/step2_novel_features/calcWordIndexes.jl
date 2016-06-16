using JSON
using Lazy
using StatsBase

include("../helpers/helpers.jl")


function allTaskFiles()
  getTaskFs(sub_dir) = getDataFile("step1", sub_dir) |> getTaskFiles
  meg_task_fs = getTaskFs("lists")
  casl_task_fs = getTaskFs("meg_sync")
  union(meg_task_fs, casl_task_fs)
end


typealias RanksDict Dict{ASCIIString, Vector{Int64}}
typealias TaskRanksDict Dict{ASCIIString, RanksDict}

typealias RankMeanStdDict Dict{ASCIIString, Tuple{Float64, Float64}}
typealias TaskRanksMeanStdDict Dict{ASCIIString, RankMeanStdDict}


isComment(w::ASCIIString) = map(c -> c in w, ('#', '!')) |> any

isValid(w::ASCIIString) = length(w) > 0 && isalpha(w[1])

function createTaskRanksDict()
  task_ranks = TaskRanksDict()

  for task_f in allTaskFiles()
    task = getTask(task_f)
    task_ranks[task] = word_ranks = get(task_ranks, task, RanksDict())

    if filesize(task_f) == 0
      continue
    end

    valid_words = begin
      all_words = readcsv(task_f, ASCIIString, comments=false)[:]
      filter(isValid, all_words)
    end

    for (rank, word) in enumerate(valid_words)
      if isComment(word)
        continue
      end

      word_ranks[word] = ranks = get(word_ranks, word, Int64[])
      push!(ranks, rank)
    end
  end

  task_ranks
end


saveDict(d::Dict, f::AbstractString) = open(f -> JSON.print(f, d), f, "w")


saveTaskRanksDict(ranks::TaskRanksDict,
  dest_f=getDataFile("step2", "ranks.json")) = saveDict(ranks, dest_f)


function convertTaskRanksToMeanStd(ranks::TaskRanksDict)
  ret = TaskRanksMeanStdDict()

  for (task, task_rank) in ranks
    ret[task] = get(ret, task, Dict())
    for (word, ranks) in task_rank
      if length(ranks) > 1
        ret[task][word] = mean_and_std(ranks)
      end
    end
  end

  ret
end


function saveTaskRanksMeanStd(ranks::TaskRanksDict,
    dest_f=getDataFile("step2", "ranks_mean_std.json"))
  saveTaskRanksMeanStd(convertTaskRanksToMeanStd(ranks), dest_f)
end


saveTaskRanksMeanStd(ranks::TaskRanksMeanStdDict,
    dest_f=getDataFile("step2", "ranks_mean_std.json")) = saveDict(
  ranks, dest_f)
