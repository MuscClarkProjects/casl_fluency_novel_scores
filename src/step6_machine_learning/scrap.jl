using DataFrames
using Lazy
using MLBase
using MLPipe
using PyCall
using UnicodePlots

include("../helpers/helpers.jl")
include("../step5_machine_learning_prep/purifyData.jl")

@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR


pipelineGen(t::Task,
  s::SummaryScore) = @> "step5" getDataCsv("$t") readtable pipelineGen(s)

function pipelineGen(df::DataFrame, s::SummaryScore)
  svr = LinearSVR()

  model_state = Dict(:svr_C => 1., :svr_penalty => "l1")
  predictors = dataCols(df)

  X_data = df[predictors] |> Matrix{Float64}
  y_data = df[symbol(s)] |> Vector{Float64}

  #fit functions
  fits = begin
    getXy(ixs) = X_data[ixs, :], y_data[ixs]

    function svrFit!(Xy)
        X, y= Xy
        svr[:C] = model_state[:svr_C]
        svr[:penalty] = model_state[:svr_penalty]
        svr[:fit](X, y)
    end

    [getXy, svrFit!]
  end

  #predict functions
  predicts = begin
    getX(ixs) = X_data[ixs, :]

    _pyArrayToJl(arr) = Float64[p for p in arr]

    svrPredict(X) = svr[:predict](X) |> _pyArrayToJl

    [getX, svrPredict]
  end

  Pipeline(fits, predicts, r2score, y_data, model_state)
end


function plotAsUni(train_scores, test_scores, title)
  y_min = minimum([train_scores; test_scores])
  y_max = maximum([train_scores; test_scores])
  p = lineplot(train_scores, name="train", title=title, ylim=[y_min, y_max])
  lineplot!(p, test_scores, name="test")
end


function evalNovelFluencyModel(t::Task, s::SummaryScore,
  svr_cs; plot=false)
  novel_fluency_pipe = pipelineGen(t, s)
  num_samples = novel_fluency_pipe.truths |> length
  cvg = RandomSub(num_samples, round(Int64, num_samples * .8), 20)

  model_eval = evalModel(novel_fluency_pipe, cvg, num_samples,
    meanTrainTest, :svr_c => svr_cs)

  if plot
    plotEvalModel(model_eval)
  end

  model_eval
end


viewPlots() = overTasks() do t::Task
  overSummaryScores() do s::SummaryScore
    model_eval = evalNovelFluencyModel(t, s, logspace(-5, 4, 10))
    title = "$t, $s"
    println(title)
    println(model_eval[1] |> maximum)
    println(model_eval[2] |> maximum)
    #plotAsUni(model_eval[1], model_eval[2], title)
  end
end
