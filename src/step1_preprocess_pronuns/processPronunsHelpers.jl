include("../helpers/helpers.jl")


pronunsDir() = getDataFile("step1", "pronunciations")

prevPronunsDir() = @>> "prev_not_in_cmu" getDataFile(pronunsDir())

newPronunsDir() = @>> "new_not_in_cmu" getDataFile(pronunsDir())


pronunsF(f, x; d=pronunsDir()) = getDataFile(d, "$(f)_pronunciations.$x")

pronunsCsv(f, d) = pronunsF(f, "csv", d=d)

pronunsCsv(f) = pronunsF(f, "csv")

prevPronunsCsv(f) = pronunsCsv(f, prevPronunsDir())

newPronunsCsv(f) = pronunsCsv(f, newPronunsCsv())


matrixToDict(data::Matrix) = Dict{ASCIIString, ASCIIString}(
  [lowercase(data[i, 1]) => data[i, 2] for i in 1:size(data, 1)]
)


readPronunsAsDict(f) = @> f pronunsCsv readcsv(ASCIIString) matrixToDict
