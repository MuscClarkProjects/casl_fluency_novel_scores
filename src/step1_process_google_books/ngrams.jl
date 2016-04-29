
validword_re = r"^[a-zA-Z|-|']+$"
valid2gram_re = r"^[a-zA-Z|-|']+ [a-zA-Z|-|']+$"
isvalid2gram(twogram) = isvalid(ASCIIString, twogram) && ismatch(valid2gram_re, twogram)
isvalid2gram{T <: AbstractString}(twogram::SubString{T}) = isvalid2gram(convert(T, twogram))


isvalidrow(leftright, count) = isvalid2gram(leftright) && ismatch(r"^[0-9]+$", count)


isvalidrow{T <: AbstractString}(cols::AbstractVector{T}) =  (length(cols) > 2) &&
  isvalidrow(cols[1], cols[3])


function get_leftright_count{T <: AbstractString}(cols::AbstractVector{T})
  convert(ASCIIString, cols[1]), parse(Int64, cols[3])
end


loaddata{T <: AbstractString}(lines::Vector{T}) = reduce(Dict{ASCIIString, Int64}(), lines) do acc, line
  cols = split(line, '\t')
  if isvalidrow(cols)
    leftright::ASCIIString, count::Int64 = get_leftright_count(cols)
    acc[leftright] = get(acc, leftright, 0) + count
  end
  acc
end


loaddata(f_name::AbstractString) = open(readlines, f_name, "r")
