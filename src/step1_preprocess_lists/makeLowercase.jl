using Lazy

include("../helpers/helpers.jl")

run(src_dir=getDataFile("step1", "lists"),
    dest_dir=getDataFile("step1", "lists_lowercase")) = for f in readdir(src_dir)
  lowercase_lines = @>> f joinpath(src_dir) readlower
  @> dest_dir joinpath(f) write(lowercase_lines)
end


readlower(stream) = @>> stream readlines map(lowercase)

readlower(f::AbstractString) = open(readlower, f)
