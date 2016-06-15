using Lazy

project_folder = dirname(dirname(pwd()))


getDataFile(parts...) = joinpath(project_folder, "data", parts...)


#####Tasks

task_reg = r"_([a-zA-Z_]+).txt"

getTask(f_name) = match(task_reg, f_name).captures[1]

getTasks(f_names) = ASCIIString[getTask(f) for f in f_names]


getTaskFiles(src_dir, extrafilter=f -> true) = @>> readdir(src_dir) begin
    filter(f -> ismatch(task_reg, f))
    filter(extrafilter)
    map(f -> joinpath(src_dir, f))
  end

#####
