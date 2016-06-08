using Logging
using GZip

include("../helpers/TwoChar.jl")


typealias Letters Union{TwoChar, Char}

validword_re = r"^[a-zA-Z|-|']+$"

function isValidGram(text::AbstractString, gram::Int64)
  if !isvalid(ASCIIString, text)
    return false
  end

  words::AbstractVector{SubString} = split(text, ' ')

  if length(words) != gram
    return false
  end

  reduce(true, words) do acc::Bool, word::AbstractString
    acc & ismatch(validword_re, word)
  end
end

isValidGram{T <: AbstractString}(text::SubString{T}, gram::Int64) = isValidGram(convert(T, text), gram)


isValidRow(text, count, gram::Int64) = isValidGram(text, gram) && ismatch(r"^[0-9]+$", count)


isValidRow{T <: AbstractString}(cols::AbstractVector{T}, gram::Int64) =  (length(cols) > 2) &&
  isValidRow(cols[1], cols[3], gram)


function getGramTextAndCount{T <: AbstractString}(cols::AbstractVector{T})
  convert(ASCIIString, lowercase(cols[1])), parse(Int64, cols[3])
end


squishCounts(lines, gram::Int64) = reduce(Dict{ASCIIString, Int64}(), lines) do acc, line
  if length(line) < 3
    return acc
  end
  cols = split(line, '\t')
  if isValidRow(cols, gram)
    text::ASCIIString, count::Int64 = getGramTextAndCount(cols)
    acc[text] = get(acc, text, 0) + count
  end
  acc
end


squishCounts(io::IO, gram::Int64) = squishCounts(eachline(io), gram)

squishCountsGen(gram::Int64) = io::IO -> squishCounts(io, gram)
squishCounts(f_name::AbstractString, gram::Int64) = open(squishCountsGen(gram), f_name, "r")


function genUrl(letters::Letters, gram::Int64)
  f_name = "googlebooks-eng-all-$(gram)gram-20120701-$(letters).gz"
  "http://storage.googleapis.com/books/ngrams/books/$(f_name)"
end


logIt(tc::Letters, msg::ASCIIString) = remotecall(1, info, "worker $(myid()): $tc $msg")

function main{T <: Letters}(gram::Int64, dest_dir::AbstractString;
  pairs::AbstractVector{T}=TwoChar('a', 'a'):TwoChar('z', 'z'))

  log_f::ASCIIString = "$gram.log"
  isfile(log_f) && rm(log_f)

  Logging.configure(filename="$(gram).log")
  Logging.configure(level=INFO)

  for tc::Letters in pairs

    gz_file_name::ASCIIString = downloadLargeGz(tc, gram, dest_dir)
    @spawn _postDownloadProcess(gz_file_name, tc, gram)
  end

end


function _postDownloadProcess(gz_file_name::AbstractString,
                              tc::Letters,
                              gram::Int64)
  g_stream::IO = GZip.gzopen(gz_file_name)
  try
    logIt(tc, "calculate counts, size: $(filesize(gz_file_name)/1e6) MB")
    counts::Dict{ASCIIString, Int64} = squishCounts(eachline(g_stream), gram)
    counts_file_name = replace(gz_file_name, ".gz", "_counts.tsv")
    writedlm(counts_file_name, counts)
    logIt(tc, "counts calculated")
  catch e
    logIt(tc, "error thrown trying to calculate counts")
    logIt(tc, e)
  finally
    close(g_stream)
    rm(gz_file_name)
  end
end


function downloadLargeGz(tc::Letters, gram::Int64, dest_dir::AbstractString,
    decompress=false)
  url::ASCIIString = genUrl(tc, gram)
  f = joinpath(dest_dir, basename(url))
  logIt(tc, "download")
  download(url, f)
  logIt(tc, "downloaded")

  if decompress
    Base.run(`gzip -df $f`)
    f_unzipped = replace(f, ".gz", "")

    new_name = "$(f_unzipped).tsv"
    mv(f_unzipped, new_name, remove_destination=true)
    new_name
  else
    f
  end
end
