using DataFrames
using Gadfly
using Lazy

using MLPipe

include("../helpers/helpers.jl")

include("purifyData.jl")


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
