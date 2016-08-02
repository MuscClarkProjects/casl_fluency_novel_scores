using DataFrames
using Lazy

include("../helpers/helpers.jl")


function readTask(task::Task, baseline_only=true)
  data = @>> "$task" getDataCsv("step4") readtable
  baseline_only ? data[data[:visit] .== "y0", :] : data
end

dataCols(task::DataFrame) = @> task names setdiff([summary_cols'; meta_cols])
dataCols(task::Task) = @> task readTask dataCols

pureData(task, all_cols=true) = @> task readTask pureData(all_cols)

function pureData(task::DataFrame, all_cols=true)
  ret = copy(task)

  cols = all_cols ? names(task) : dataCols(task)

  pure_rows = @> task[:, cols] complete_cases

  for c in dataCols(task)
    vec = ret[pure_rows, c]

    if reduce(|, isinf(vec))
      warn("inf found in $c")
      warn("will flip $c")

      println("## $c info ##")
      println("max: $(maximum(vec)); min: $(minimum(vec))")
      println("min abs: $(minimum(abs(vec)))")
      println("####")

      new_col = symbol("one_over_", c)
      ret[new_col] = 0.
      ret[pure_rows, new_col] = 1./vec
      delete!(ret, c)
    end

    if std(vec) == 0.0
      warn("std = 0 in $c, will delete")
      delete!(ret, c)
    end

  end

  ret[pure_rows, :]
end


writePureData!() = overTasks() do t::Task
  @>> t pureData writetable(getDataCsv("step5", "$t"))
end
