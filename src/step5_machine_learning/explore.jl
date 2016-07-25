using DataFrames
using Lazy

include("../helpers/helpers.jl")


function readTask(task, baseline_only=true)
  data = @>> task getDataCsv("step4") readtable
  baseline_only ? data[data[:visit] .== "y0", :] : data
end


pureData(task, include_y=true) = @> task readTask pureData(include_y)

function pureData(task::DataFrame, include_y=true)
  data_cols = if include_y
    names(task)
  else
    @> task names setdiff([summary_cols; :id])
  end

  pure_rows = @> task[:, data_cols] complete_cases

  task[pure_rows, :]
end
