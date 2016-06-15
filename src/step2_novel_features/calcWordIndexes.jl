using JSON
using Lazy

include("../helpers/helpers.jl")


function allTaskFiles()
  getTaskFs(sub_dir) = getDataFile("step1", sub_dir) |> getTaskFiles
  meg_task_fs = getTaskFs("lists")
  casl_task_fs = getTaskFs("meg_sync")
  union(meg_task_fs, casl_task_fs)
end


typealias IndexesDict Dict{ASCIIString, Vector{Int64}}
typealias RanksDict Dict{ASCIIString, IndexesDict}


isComment(w::ASCIIString) = map(c -> c in w, ('#', '!')) |> any

isValid(w::ASCIIString) = length(w) > 0 && isalpha(w[1])

function createRanksDict()
  ranks = RanksDict()

  for task_f in allTaskFiles()
    task = getTask(task_f)
    ranks[task] = get(ranks, task, IndexesDict())

    if filesize(task_f) == 0
      continue
    end

    valid_words = begin
      all_words = readcsv(task_f, ASCIIString, comments=false)[:]
      filter(isValid, all_words)
    end

    for (w_ix, word) in enumerate(valid_words)
      if isComment(word)
        continue
      end

      ranks[task][word] = get(ranks[task], word, ASCIIString[])
      push!(ranks[task][word], w_ix)
    end
  end

  ranks
end


saveRanksDict(ranks::RanksDict,
  dest_f=getDataFile("step2", "ranks.json")) = open(f -> JSON.print(f, ranks),
    dest_f, "w")
