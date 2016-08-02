using DataFrames
using Lazy

include("../helpers/helpers.jl")


function readTask(task::Task, baseline_only=true)
  data = @>> "$task" getDataCsv("step4") readtable
  baseline_only ? data[data[:visit] .== "y0", :] : data
end

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


function writePureData!()
  writeStep5(f_name, data) = @> "step5" getDataCsv(f_name) writetable(data)

  pure_data_tasks = overTasks() do t::Task
    pure_data = pureData(t)
    writeStep5("$t", pure_data)

    pure_data
  end

  all_tasks = mergeTasks(Tasks())
  writeStep5("all_tasks", all_tasks)
end


mergeTasks(tasks) = reduce(DataFrame(), tasks) do acc::DataFrame, t::Task
  df = begin
    task_df = pureData(t)

    @>> [:id] setdiff(meta_cols) delete!(task_df)

    rename!(task_df,
      [c => symbol(c, '_', t) for c in dataCols(task_df)]
    )

    task_df
  end

  is_first = (@> acc names length) == 0

  if is_first
    df
  else
    non_summary_cols = @> df names setdiff(summary_cols)

    ret = join(acc, df[non_summary_cols], on=:id)

    df_id_ixs, acc_id_ixs = map([df, acc]) do d
      Int64[findfirst(d[:id], i) for i in ret[:id]]
    end

    assert(
      @>> summary_cols begin
        map(c -> df[df_id_ixs, c] == acc[acc_id_ixs, c])
        reduce(&)
      end
    )
    ret
  end
end
