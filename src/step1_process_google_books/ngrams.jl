using Logging

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


squishCounts(io::IOStream, gram::Int64) = squishCounts(eachline(io), gram)

squishCountsGen(gram::Int64) = io -> squishCounts(io, gram)
squishCounts(f_name::AbstractString, gram::Int64) = open(squishCountsGen(gram), f_name, "r")


function genUrl(left::Char, right::Char, gram::Int64)
  "http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-$(gram)gram-20120701-$(left)$(right).gz"
end


type TwoChar
  left::Char
  right::Char

  TwoChar(left, right) = begin
    for c::Char in (left, right)
      islower(c) || error("$c must be lowercase alpha-numeric")
    end
    new(left, right)
  end
end


genUrl(tc::TwoChar, gram::Int64) = genUrl(tc.left, tc.right, gram)


Base.convert(::Type{Int64}, tc::TwoChar) = begin
  val(c::Char) = Int(c) - Int('a')
  26*val(tc.left) + val(tc.right)
end


function +(tc::TwoChar, bias::Int64)
  if sign(bias) < 0
    return tc - abs(bias)
  end
  right::Char = tc.right + bias%26
  left::Char =  tc.left + floor(Int64, bias/26)
  is_rollover::Bool = right > 'z'
  if is_rollover
    left += 1
    right -= 26
  end
  TwoChar(left, right)
end


+(tc1::TwoChar, tc2::TwoChar) = tc1 + Int(tc2)
-(tc1::TwoChar, tc2::TwoChar) = tc1 - Int(tc2)
Base.isless(tc1::TwoChar, tc2::TwoChar) = Int(tc1) < Int(tc2)
Base.one(::TwoChar) = TwoChar('a', 'b')
Base.one(::Type{TwoChar}) = one(TwoChar)
Base.zero(::TwoChar) = TwoChar('a', 'a')
Base.zero(::Type{TwoChar}) = TwoChar('a', 'a')
Base.rem(num::TwoChar, denom::TwoChar) = zero(TwoChar) + Int(num)%Int(denom)


function -(tc::TwoChar, bias::Int64)
  if sign(bias) < 0
    return tc + abs(bias)
  end
  right::Char = tc.right - bias%26
  left::Char =  tc.left - floor(Int64, abs(bias)/26)
  is_rollover::Bool = right < 'a'
  if is_rollover
    left -= 1
    right += 26
  end
  TwoChar(left, right)
end


function main(gram::Int64, dest_dir::AbstractString; 
  start_from::TwoChar=TwoChar('a', 'a'), 
  run_until::TwoChar = TwoChar('z', 'z'))

  log_f::ASCIIString = "$gram.log"
  isfile(log_f) && rm(log_f)

  Logging.configure(filename="$(gram).log")
  Logging.configure(level=INFO)

  pmap(start_from:run_until) do tc::TwoChar
    logIt(msg::ASCIIString) = remotecall(1, info, msg)
  
    url::ASCIIString = genUrl(tc, gram)
    
    logIt("downloading $tc")
    f::AbstractString = downloadLargeFile(url, dest_dir)
    logIt("$tc downloaded")
    
    logIt("calculating $tc counts")
    counts::Dict{ASCIIString, Int64} = squishCounts(f, gram)
    
    f_counts = replace(f, ".tsv", "_counts.tsv")
    writedlm(f_counts, counts)
    
    logIt("$tc counts calculated")
    
    rm(f)
  end

end


function downloadLargeFile(url, dest_dir::AbstractString)
  f = joinpath(dest_dir, basename(url))
  println("downloading $f")
  download(url, f)
  println("downloaded $f")

  Base.run(`gzip -df $f`)
  f_unzipped = replace(f, ".gz", "")

  new_name = "$(f_unzipped).tsv"
  mv(f_unzipped, new_name, remove_destination=true)
  new_name
end
