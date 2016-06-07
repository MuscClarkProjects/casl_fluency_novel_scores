using DataFrames
using JSON
using Lazy
using Logging

include("../helpers/helpers.jl")
include("../helpers/TwoChar.jl")


readStringsFile(f_path) = readcsv(f_path, ASCIIString)[:]

function targetWords()
  target_words_f = getDataFile("step1", "targetwords", "target_words.txt")
  ret::AbstractVector{ASCIIString} = readStringsFile(
    target_words_f)

  multispellings_f = getDataFile("step1", "targetwords", "multispellings.txt")
  append!(ret, readStringsFile(multispellings_f))
  ASCIIString[replace(t, '_', ' ') for t in ret]
end

function getCounts(ngram::Int64, tc::TwoChar, db_dir::AbstractString)
  fname = joinpath(db_dir,
    "googlebooks-eng-all-$(ngram)gram-20120701-$(tc)_counts.tsv")
  if filesize(fname) > 0
    readtable(fname, header=false, names=[:gram, :count])
  else
    DataFrame(gram=[], count=[])
  end
end

immutable ContextRow
  context::AbstractString
  target::AbstractString
  tp::Symbol
  count::Int64
end


hasStopWordEdge(words, stop_words::Set{ASCIIString}=stopWords()) =
  (words[1] in stop_words) |
  (words[end] in stop_words)


function createContextRow(gram::ASCIIString, count::Int64,
  target_words::AbstractVector{ASCIIString} = targetWords(),
  stop_words::Set{ASCIIString} = stopWords())

  words = split(gram, ' ')

  left_target::Nullable{ContextRow} = Nullable{ContextRow}()
  right_target::Nullable{ContextRow} = Nullable{ContextRow}()

  if hasStopWordEdge(words, stop_words)
    return left_target, right_target
  end

  is_left_target = is_right_target = false
  for t in target_words
    if ' ' in t
      ts = split(t, ' ')
      num_ts = length(ts)
      if num_ts >= length(words)
        continue
      end

      isX(get_word) = reduce(true, 1:num_ts) do acc::Bool, i::Int64
        acc & (ts[i] == get_word(i))
      end

      is_left_target = isX(i::Int64 -> words[i])
      is_right_target = isX(i -> words[end - num_ts + i])
    else
      is_left_target = words[1] == t
      is_right_target = words[end] == t
    end

    if is_left_target
      left_target = Nullable(ContextRow(words[end], t, :left, count))
    end
    if is_right_target
      right_target = Nullable(ContextRow(words[1], t, :right, count))
    end
  end

  left_target, right_target
end


createContextRow{S <: AbstractString}(gram::S, count::Int64,
  target_words::AbstractVector{ASCIIString} = targetWords(),
  stop_words::Set{ASCIIString} = stopWords()
  ) = if !isvalid(ASCIIString, gram)
    Nullable{ContextRow}(), Nullable{ContextRow}()
  else
    createContextRow(ASCIIString(gram), count, target_words, stop_words)
end


function createContextRow(row::DataFrameRow,
  target_words::AbstractVector{ASCIIString}=targetWords(),
  stop_words::Set{ASCIIString}=stopWords())

  createContextRow(row[:gram], row[:count], target_words, stop_words)
end


logIt(tc::TwoChar, msg::ASCIIString) = remotecall(
  1, info, "worker $(myid()): $tc $msg")


function createContextRows(ngram::Int64, tc::TwoChar, db_dir::AbstractString)
  counts::DataFrame = getCounts(ngram, tc, db_dir)
  num_rows = size(counts, 1)

  stop_words = stopWords()
  target_words = targetWords()
  ret = ContextRow[]
  prev_ix = 1
  for (ix::Int64, row::DataFrameRow) in enumerate(eachrow(counts))
    if ix - prev_ix > num_rows/10
      prev_ix = ix
      logIt(tc, "$ix out of $(num_rows) done")
    end

    left_cr::Nullable{ContextRow}, right_cr::Nullable{ContextRow} =
      createContextRow(row, target_words, stop_words)

    map((left_cr, right_cr)) do cr
      isnull(cr) || (push!(ret, get(cr)))
    end
  end

  logIt(tc, "calculated context rows")

  ret
end


function process(ngram::Int64, tc::TwoChar,
  db_dir::AbstractString, dest_dir::AbstractString)

  data = JSON.json(createContextRows(ngram, tc, db_dir))
  outfile = open(joinpath(dest_dir, "$(tc)_context.json"), "w")
  write(outfile, data)
  close(outfile)

  logIt(tc, "processing complete")
end


function main(ngram::Int64, dest_dir::AbstractString, db_dir::AbstractString;
  pairs::AbstractVector{TwoChar}=TwoChar('a', 'a'):TwoChar('z', 'z'))

  log_f::ASCIIString = "$ngram.log"
  isfile(log_f) && rm(log_f)

  Logging.configure(filename="$(ngram).log")
  Logging.configure(level=INFO)

  pmap(pairs) do tc::TwoChar
    process(ngram, tc, db_dir, dest_dir)
  end

end


function dirToTwoChars(reg::Regex, dir_path)
  ret = @>> readdir(dir_path) begin
    map(s -> match(reg, s))
    filter(m -> m !== nothing)
    map(m -> TwoChar(m.captures[1][1], m.captures[1][2]))
  end
  TwoChar[r for r in ret]
end

destTs(dest_dir) = dirToTwoChars(r"^(\w\w)", dest_dir)
dbTs(db_dir) = dirToTwoChars(r"(\w\w)_counts", db_dir)


function resumeMain(ngram::Int64, dest_dir, db_dir)
  ts = setdiff(dbTs(db_dir), destTs(dest_dir))
  main(ngram, dest_dir, db_dir, pairs=ts)
end


stopWords() = Set(readcsv(getDataFile("step1", "stopwords", "stopWords.txt"),
                      ASCIIString,
                      header=false)[:])


disallowed(w) = any(l -> l in w, "!@#\$%^&*(){}1234567890=+[]']"])
