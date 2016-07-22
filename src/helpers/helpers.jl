using Lazy

project_folder = dirname(dirname(pwd()))


getDataFile(parts...) = joinpath(project_folder, "data", parts...)


getDataCsv(parts...) = @> getDataFile(parts...) string(".csv")

#####Tasks

task_reg = r"_([a-zA-Z_]+).txt"

getTask(f_name) = match(task_reg, f_name).captures[1]

getTasks(f_names) = ASCIIString[getTask(f) for f in f_names]


getTaskFiles(src_dir, extrafilter=f -> true) = @>> readdir(src_dir) begin
    filter(f -> ismatch(task_reg, f))
    filter(extrafilter)
    map(f -> joinpath(src_dir, f))
  end


function allTaskFiles()
  getTaskFs(sub_dir) = getDataFile("step1", sub_dir) |> getTaskFiles
  meg_task_fs = getTaskFs("lists")
  casl_task_fs = getTaskFs("meg_sync")
  union(meg_task_fs, casl_task_fs)
end

#####

saveDict(d::Dict, f::AbstractString) = open(f -> JSON.print(f, d), f, "w")


#####Words helpers
isComment(w::ASCIIString) = map(c -> c in w, ('#', '!')) |> any

isValidWord(w::ASCIIString) = length(w) > 0 && isalpha(w[1])

#####
