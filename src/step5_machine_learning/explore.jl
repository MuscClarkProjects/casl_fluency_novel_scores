using DataFrames
using Lazy

include("../helpers/helpers.jl")


function readTask(task, baseline_only=true)
  data = @>> task getDataCsv("step4") readtable
  baseline_only ? data[data[:visit] .== "y0", :] : data
end


allNas(task, include_y=true) = @> task readTask allNas(include_y)

function allNas(task::DataFrame, include_y=true)
  data_cols = include_y ? names(task) : @> task names setdiff([summary_cols; :id])

  na_rows = reduce(Vector{Int64}(), data_cols) do v, c::Symbol
    [v; task[c] |> isna |> find]
  end

  @>> na_rows setdiff(1:size(task, 1)) tuple(na_rows)
end


pureData(task, include_y=true) = @> task readTask pureData(include_y)

function pureData(task::DataFrame, include_y=true)
  pure_rows = @> task allNas(include_y) getindex(2)
  task[pure_rows, :]
end
