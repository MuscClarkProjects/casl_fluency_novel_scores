using DataFrames
using Lazy

typealias LogUni Dict{ASCIIString, Float64}

function addOneGramToLogUni!(onegrams_f::AbstractString, log_uni::LogUni)
  grams = readtable(onegrams_f, header=false, names=[:word, :count])
  for g_row::DataFrameRow in eachrow(grams)
    word = g_row[:word]
    count = g_row[:count]
    log_uni[word] = log(count)
  end

  log_uni
end


function main(onegrams_dir::AbstractString, dest_f::AbstractString)
  onegram_fs = @>> readdir(onegrams_dir) map(f -> joinpath(onegrams_dir, f))
  log_uni = LogUni()

  for onegram_f in onegram_fs
    println("about to process $(onegram_f)")
    addOneGramToLogUni!(onegram_f, log_uni)
  end

  open(f -> JSON.print(f, log_uni), dest_f, "w")
end


rawProb(logunigrams_f::AbstractString) = rawProb(
  JSON.parsefile(logunigrams_f, dicttype=StringFloatMap))


function rawProb(logunigrams::StringFloatMap)
  raws = StringFloatMap([ k => exp(raw) for (k, raw) in logunigrams ])
  total_raw = raws |> values |> sum
  StringFloatMap([key => raw/total_raw for (key, raw) in raws])
end
