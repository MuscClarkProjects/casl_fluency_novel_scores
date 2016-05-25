using DataFrames
using Lazy
using Requests

#http://storage.googleapis.com/books/ngrams/books/datasetsv2.html

immutable NGramUrlMap
  gram_1::AbstractString
  gram_2::AbstractString
  gram_3::AbstractString
  gram_4::AbstractString
  gram_5::AbstractString
end


get_url(word_comp::AbstractString, gram_count::Int64) = begin
  "http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-$(gram_count)gram-20120701-$(word_comp).gz"
end


get(ng::NGramUrlMap, gram::Int64) = ng.(symbol("gram_", gram))


function NGramUrlMap(word::AbstractString)
  word_comp::AbstractString = length(word) > 2 ? word[1:2] : "$(word)_",

  NGramUrlMap([get_url(word_comp, gc) for gc in 1:5]...)
end

NGramUrlMap(word::Integer) = begin
  @assert word > -1 & word < 10

  NGramUrlMap("$word")
end


function checkdb(dbdir::AbstractString, word)
end


load_rawcounts(f::AbstractString) = readtable(
  f, names=[:pair, :year, :match_count, :volume_count])


function aggregatecounts(df::DataFrame)
  function sumcounts(d)
    left::AbstractString, right::AbstractString = split(d[:pair][1], " ")
    DataFrame(left=left, right=right, match_count=sum(d[:match_count]))
  end
  by(df, :pair, sumcounts)
end


aggregatecounts(df_file::AbstractString) = aggregatecounts(load_rawcounts(df_file))


function run(counts_dbdir, pmi_des, pmi_orig::Dict)
end


function run(counts_dbdir, pmi_des, pmi_orig::AbstractString)
  run(counts_dbdir, pmi_des, JSON.parsefile(pmi_orig))
end
