
validword_re = r"^[a-zA-Z|-|']+$"
validpair_re = r"^[a-zA-Z|-|']+ [a-zA-Z|-|']+$"
isvalidword(word) = isvalid(ASCIIString, word) && ismatch(validword_re, word)
isvalidword{T <: AbstractString}(word::SubString{T}) = isvalidword(convert(T, word))


function isvalidrow(left, right, count)
  isvalidword(left) && isvalidword(right) && ismatch(r"^[0-9]+$", count)
end


function isvalidrow{T <: AbstractString}(cols::AbstractVector{T})
  if (length(cols) > 2) && contains(cols[1], " ")
    left::AbstractString, right::AbstractString = split(cols[1], ' ')
    isvalidrow(left, right, cols[3])
  else
    false
  end
end


function get_leftright_count{T <: AbstractString}(cols::AbstractVector{T})
  left, right = split(cols[1], ' ')
  count = cols[3]

  (convert(ASCIIString, left), convert(ASCIIString, right)), parse(Int64, count)
end


typealias WordPair Tuple{ASCIIString, ASCIIString}
loaddata{T <: AbstractString}(lines::Vector{T}) = reduce(Dict{WordPair, Int64}(), lines) do acc, line
  cols = split(line, '\t')
  if isvalidrow(cols)
    leftright::WordPair, count::Int64 = get_leftright_count(cols)
    acc[leftright] = get(acc, leftright, 0) + count
  end
  acc
end


loaddata(f_name::AbstractString) = open(readlines, f_name, "r")
