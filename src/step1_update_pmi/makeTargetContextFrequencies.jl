using DataFrames
using JSON
using Lazy


include("../helpers/helpers.jl")


typealias ContextFrequency Dict{ASCIIString, Int64}
typealias TargetFrequency Dict{ASCIIString, ContextFrequency}


function main(context_db_dir::AbstractString, dest_f::AbstractString,
    target_words::AbstractString=getDataFile(
      "step1", "targetwords", "target_words.txt"))
  main(context_db_dir, dest_f, readcsv(target_words, ASCIIString)[:])
end


function main(context_db_dir::AbstractString,
              dest_f::AbstractString,
              target_words::Vector{ASCIIString})

  context_dirs = map(d -> joinpath(context_db_dir, "$(d)grams"), 2:5)
  target_freq::TargetFrequency = [t => ContextFrequency() for t in target_words]

  for context_dir in context_dirs
    addContextDirToTargetFrequency!(context_dir, target_freq)
  end

  open(f -> JSON.print(f, target_freq), dest_f, "w")

  target_freq
end


function addContextDirToTargetFrequency!(context_dir::AbstractString,
                                          target_freq::TargetFrequency)
  context_fs = @>> readdir(context_dir) map(f -> joinpath(context_dir, f))
  for context_f in context_fs
    println("about to process $(context_f)")
    addContextFileToTargetFrequency!(context_f, target_freq)
  end

  target_freq
end


function addContextFileToTargetFrequency!(context_f::AbstractString,
    target_freq::TargetFrequency)

  contexts = JSON.parsefile(context_f)
  for c::Dict in contexts
    target::ASCIIString = c["target"]

    if haskey(target_freq, target)
      context::ASCIIString = c["context"]

      context_freq::ContextFrequency = target_freq[target]

      count::Int64 = get(context_freq, context, 0) + c["count"]
      target_freq[target][context] = count
    end

  end

  target_freq
end
