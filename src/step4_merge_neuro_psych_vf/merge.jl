using DataFrames
using Lazy

include("../helpers/helpers.jl")


function summaryScores()
  df = readtable(getDataCsv("step3", "summary_scores"))

  df[:int_id] = Int64[parse(Int64, s[end-1:end]) for s in df[:subject]]

  delete!(df, :subject)
  rename!(df, :int_id, :id)

  baseline_rows::Vector{Bool} = begin
    same_date = df[:baseline_date] .== df[:date]
    is_na = [isna(d) for d in df[:baseline_date]]
    same_date | is_na
  end

  df[baseline_rows, [:id; summary_cols]]
end


function run()
  summary_scores = summaryScores()

  const step2_dir = getDataFile("step2")
  for task_f in @>> step2_dir readdir filter(f -> endswith(f, "txt"))
    vf = @> step2_dir joinpath(task_f) readtable(separator='\t',
      nastrings=["None"]);

    dest_f = @>> task_f[6:end-4] getDataCsv("step4")

    @>> join(vf, summary_scores[[:id; summary_cols]], on=:id) writetable(dest_f)
  end

end
