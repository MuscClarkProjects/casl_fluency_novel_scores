using DataFrames
using Gadfly
using Lazy

using MLPipe

include("../helpers/helpers.jl")


function readTask(task::Task, baseline_only=true)
  data = @>> "$task" getDataCsv("step4") readtable
  baseline_only ? data[data[:visit] .== "y0", :] : data
end

const meta_cols = [:id, :visit, :task]
const summary_cols = overSummaryScores(symbol)

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


safePlots(plots) = filter(plots) do p
  try
    display(p)
    true
  catch e
    warn(p.guides)
    false
  end
end


plotFeature(task::DataFrame, predictor, prediction) = plot(task,
  x=predictor,
  y=prediction,
  Geom.point,
  Geom.smooth,
  Guide.xlabel("$predictor"),
  Guide.ylabel("$prediction")
  )


function plotFeatures(task_name)
  task = pureData(task_name)

  predictors = dataCols(task)
  for prediction in summary_cols
    println("will run for $prediction")

    plots = begin
      all_plots = [plotFeature(task, c, prediction) for c in predictors]

      safePlots(all_plots)
    end

    pic_len = begin
      num_rows = length(predictors)
      parse("$(num_rows*5)inch") |> eval
    end

    draw(PNG("$(task_name)_$(prediction).png", 9inch, pic_len), vstack(plots))
  end
end


function calcCorr(task::Task, pred)
  data = @> task readTask pureData
  @> data calcCorrelations(dataCols(data), symbol(pred))
end


function calcCorrs(create_plot=true)
  sum_corrs = overSummaryScores() do s::SummaryScore
    println("will run for $s")

    task_corrs = overTasks() do t::Task
      corr = calcCorr(t, s)
      corr[:task] = "$t"
      corr[:score] = "$s"
      corr
    end

    vcat(task_corrs...)
  end

  main_corrs = vcat(sum_corrs...)

  if create_plot
    p = plot(main_corrs, xgroup="task", ygroup="score", x="cor",
      Guide.xlabel("corr"), Guide.ylabel("count"),
      Geom.subplot_grid(Geom.histogram))

    draw(PNG("corrs.png", 15inch, 10inch), p)
  end

  main_corrs
end
