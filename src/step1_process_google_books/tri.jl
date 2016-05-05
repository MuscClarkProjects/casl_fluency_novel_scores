function toString(tri::Tri)
  ks::AbstractVector{Char} = collect(keys(tri))
  
  num_letters = length(ks)
  
  if num_letters == 1
    k::Char = ks[1]
    v = tri[k]
    is_count_key = k == '#'
    if is_count_key
      return ":{'#':$v}"
    else
      return "$k$(toString(v))"
    end
  end
   
  children::Vector{String} = map(ks) do k::Char
    v = tri[k]::Union{Tri, Int64}
    isa(v, Int64) ? "'#':$v" : "$k$(toString(v))"
  end 
  
  ":{$(join(children, ','))}"
end

typealias Tri Dict{Char, Union{Dict, Int64}}
calcTri(paircounts::Dict{ASCIIString, Int64}) = reduce(Tri(), paircounts) do acc, paircount
  wordpair::ASCIIString, count::Int64 = paircount

  num_chars = length(wordpair)
  function addToTri(tri::Tri=acc, char_ix::Int64=1)
    char::Char = wordpair[char_ix]

    if char_ix == num_chars
      count_tri = get(tri, char, Tri())
      count_tri['#'] = count
      tri[char] = count_tri
    else
      in(char, keys(tri)) || (tri[char] = Tri())
      addToTri(tri[char], char_ix + 1)
    end
  end

  addToTri()
  acc
end


function calcTri(count_f::ASCIIString)
  paircounts = readdlm(count_f, '\t')
  wordpairs = ASCIIString[p for p in paircounts[:, 1]]
  counts = Int64[c for c in paircounts[:, 2]]
  calcTri(Dict(zip(wordpairs, counts)))
end