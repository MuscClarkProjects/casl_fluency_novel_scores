using DataFrames
using Lazy


include("processPronunsHelpers.jl")


function readPronunsAsTable(f)
  ret = readtable(prevPronunsCsv(f), header=false, names=[:word, :pronun])

  ret[:word] = [lowercase(w) for w in ret[:word]]
  ret
end


pronunTypes() = ["a", "animals", "f", "fruits_and_veg", "s"]


function findInconistent()
  a, animals, f, fruits_and_veg, s = map(readPronunsAsTable, pronunTypes())

  combine(df1, df2) = join(df1, df2, on=:word, kind=:outer)
  all_df = @> a combine(animals) combine(f) combine(fruits_and_veg) combine(s)

  rename!(all_df,
    Dict(:pronun => :a,
         :pronun_1 => :animals,
         :pronun_2 => :f,
         :pronun_3 => :fruits_and_veg,
         :pronun_4 => :s))

  conflict_rows::Vector{Bool} = begin
    pronun_types = map(symbol, pronunTypes())

    map(eachrow(all_df)) do row
      @>> pronun_types begin
        map(p -> row[p])
        filter(i -> !isna(i))
        unique
        ws -> length(ws) > 1
      end
    end
  end

  all_df[conflict_rows, :]
end


readPrevPronunsAsDict(f) = @> f prevPronunsCsv readcsv(ASCIIString) matrixToDict


function readAllPronunsAsDict()
  ret::Dict = @>> map(readPrevPronunsAsDict, pronunTypes())... union Dict

  for (word, pronun) in readPronunsAsDict("prev_resolve_conflict")
    ret[word] = pronun
  end
  ret
end


writeCombined() = @> "prev_not_in_cmu" pronunsCsv writecsv(readAllPronunsAsDict())
