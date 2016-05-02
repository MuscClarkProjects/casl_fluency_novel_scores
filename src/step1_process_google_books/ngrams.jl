using Logging

validword_re = r"^[a-zA-Z|-|']+$"
validbigram_re = r"^[a-zA-Z|-|']+ [a-zA-Z|-|']+$"

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


isValidRow(leftright, count, gram::Int64) = isValidGram(leftright, gram) && ismatch(r"^[0-9]+$", count)


isValidRow{T <: AbstractString}(cols::AbstractVector{T}, gram::Int64) =  (length(cols) > 2) &&
  isValidRow(cols[1], cols[3], gram::Gram)


function getLeftrightCount{T <: AbstractString}(cols::AbstractVector{T})
  convert(ASCIIString, lowercase(cols[1])), parse(Int64, cols[3])
end


squishCounts(lines, gram::Int64) = reduce(Dict{ASCIIString, Int64}(), lines) do acc, line
  cols = split(line, '\t')
  if isValidRow(cols, gram)
    leftright::ASCIIString, count::Int64 = getLeftrightCount(cols)
    acc[leftright] = get(acc, leftright, 0) + count
  end
  acc
end


squishCounts(io::IOStream, gram::Int64) = squishCounts(eachline(io), gram)

squishCounts(f_name::AbstractString) = open(squishCounts, f_name, "r")


function genUrl(left::Char, right::Char, gram::Int64)
  "http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-$(gram)gram-20120701-$(left)$(right).gz"
end


function main(gram::Int64, dest_dir::AbstractString, start_from::Int64=1)

  log_f::ASCIIString = "$gram.log"
  isfile(log_f) && rm(log_f)

  Logging.configure(filename="$(gram).log")
  Logging.configure(level=INFO)

  a_through_z::Vector{Char} = map(Char, 97:(97+25))

  current_ix::Int64 = 0
  for left::Char in a_through_z
    for right::Char in a_through_z
      current_ix += 1
      if current_ix < start_from
        continue
      end
      url = genUrl(left, right, gram)
      info("processing $(current_ix): $url")

      f::AbstractString = downloadLargeFile(url, dest_dir)
      println(f)
      counts::Dict{ASCIIString, Int64} = squishCounts(f, gram)
      f_counts = replace(f, ".tsv", "_counts.tsv")
      writedlm(f_counts, counts)
      rm(f)

    end
  end

end


function downloadLargeFile(url, dest_dir::AbstractString)
  f = joinpath(dest_dir, basename(url))
  println("downloading $f")
  download(url, f)
  println("downloaded $f")

  Base.run(`gzip -d $f`)
  f_unzipped = replace(f, ".gz", "")

  println("f is still $(f_unzipped)")

  new_name = "$(f_unzipped).tsv"
  mv(f_unzipped, new_name)
  new_name
end
