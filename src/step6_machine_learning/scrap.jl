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
SVR = svm.SVR

@pyimport sklearn.ensemble as ens
RandomForestRegressor = ens.RandomForestRegressor


function rfPipelineGen(t, s::SummaryScore)
  model_state = Dict{Symbol, Any}()
  classifier = model_state[:classifier] = RandomForestRegressor()
  pipelineGen(t, s, classifier, model_state)
end

function svrPipelineGen(t, s::SummaryScore)
  model_state = Dict(:C => 1., :penalty => "l1", :kernel => "rbf")
  classifier = model_state[:classifier] = SVR()
  pipelineGen(t, s, classifier, model_state,
  fit_keys = Symbol[m for m in keys(model_state)]
  )
end

function linearSvrPipelineGen(t, s::SummaryScore)
  model_state = Dict(:C => 1., :penalty => "l1")
  classifier = model_state[:classifier] = LinearSVR()
  pipelineGen(t, s, classifier, model_state)
end


step5TaskDf(t) = @> "step5" getDataCsv("$t") readtable

pipelineGen(t,
  s::SummaryScore,
  classifier,
  model_state::Dict;
  fit_keys::Vector{Symbol} = intersect(keys(model_state), keys(classifier))
  ) = @> t step5TaskDf pipelineGen(s, classifier, model_state, fit_keys=fit_keys)

function pipelineGen(df::DataFrame, s::SummaryScore,
  classifier, model_state::Dict;
  fit_keys = intersect(keys(model_state), keys(classifier))
  )

  predictors = dataCols(df)

  X_data = df[predictors] |> Matrix{Float64}
  y_data = df[symbol(s)] |> Vector{Float64}

  #fit functions
  fits = begin
    getXy(ixs) = X_data[ixs, :], y_data[ixs]

    function classifierFit!(Xy)
      for k in fit_keys
        classifier[k] = model_state[k]
      end

      X, y = Xy
      classifier[:fit](X, y)
    end

    [getXy, classifierFit!]
  end

  #predict functions
  predicts = begin
    getX(ixs) = X_data[ixs, :]

    _pyArrayToJl(arr) = Float64[p for p in arr]

    classifierPredict(X) = classifier[:predict](X) |> _pyArrayToJl

    [getX, classifierPredict]
  end

  Pipeline(fits, predicts, r2score, y_data, model_state)
end


function plotAsUni(train_scores, test_scores, title)
  y_min = minimum([train_scores; test_scores])
  y_max = maximum([train_scores; test_scores])
  p = lineplot(train_scores, name="train", title=title, ylim=[y_min, y_max])
  lineplot!(p, test_scores, name="test")
end


function pipelineToRandomSub(p::Pipeline)
  num_subjects = length(p.truths)
  sample_length = round(Int64, num_subjects * .8)
  num_samples = 20
  RandomSub(num_subjects, sample_length, num_samples)
end


function evalNovelFluencyModel(pipe::Pipeline,
  params::Vector{Pair}; plot=false)

  num_samples = pipe.truths |> length
  cvg = pipelineToRandomSub(pipe)

  model_eval = evalModel(pipe, cvg, num_samples, meanTrainTest, params...)

  if plot
    plotEvalModel(model_eval)
  end

  model_eval
end


viewPlots(;tasks = Tasks(),
  summary_scores=summaryScores()) = overTasks(tasks) do t
    overSummaryScores(summary_scores) do s::SummaryScore
      function printModelEval(model_eval, model_type)
        title = "$(model_type): $t, $s"
        println(title)
        println(model_eval[1] |> maximum)
        println(model_eval[2] |> maximum)
        #plotAsUni(model_eval[1], model_eval[2], title)
      end
      lin_svr_eval = begin
        lin_svr_pipe = linearSvrPipelineGen(t, s)
        evalNovelFluencyModel(lin_svr_pipe, Pair[:C=>logspace(-5, 4, 10)])
      end
      printModelEval(lin_svr_eval, "Linear SVR")

      svr_eval = begin
        svr_pipe = svrPipelineGen(t, s)
        evalNovelFluencyModel(svr_pipe, Pair[:C=>logspace(-5, 4, 10)])
      end
      printModelEval(svr_eval, "SVR")

      rf_eval = begin
        rf_pipe = rfPipelineGen(t, s)
        evalNovelFluencyModel(svr_pipe, Pair[:n_features=>[10, 15, 20]])
      end
      printModelEval(rf_eval, "Random Forests")
    end
end
