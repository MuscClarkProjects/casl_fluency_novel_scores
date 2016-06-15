using Lazy

include("../helpers/helpers.jl")

typealias MegToCaslMap Dict{ASCIIString, ASCIIString}

task_mapping = MegToCaslMap(
  "a"=>"a",
  "animal"=>"animals",
  "aquatic"=>"water_creatures",
  "boat" => "boats",
  "f" => "f",
  "fruitveg"=>"fruits_and_veg",
  "s" =>"s",
  "tool"=>"tools",
  "vehicle"=>"vehicles",
  "verb" => "verbs"
)


task_reg = r"_([a-zA-Z_]+).txt"

getTask(f_name) = match(task_reg, f_name).captures[1]

getTasks(f_names) = ASCIIString[getTask(f) for f in f_names]


getTaskFiles(src_dir, extrafilter=f -> true) = @>> readdir(src_dir) begin
    filter(f -> ismatch(task_reg, f))
    filter(extrafilter)
    map(f -> joinpath(src_dir, f))
  end


getMegTaskFiles() = getTaskFiles(getDataFile("step1", "meg_vf"),
  f -> startswith(f, 'M'))


getCaslTaskFiles() = getTaskFiles(getDataFile("step1", "lists"))


function synchronize(megdest_dir=getDataFile("step1", "meg_sync"))
  isdir(megdest_dir) && rm(megdest_dir, recursive=true)
  mkdir(megdest_dir)

  meg_task_fs = getMegTaskFiles()
  for meg_f in meg_task_fs
    dest_f = begin
      meg_task = getTask(meg_f)
      casl_task = task_mapping[meg_task]
      meg_f_name = basename(meg_f)
      dest_f_name = replace(meg_f_name, meg_task, casl_task)
      joinpath(megdest_dir, dest_f_name)
    end

    cp(meg_f, dest_f)
  end
end


function checkSync(megsync_dir=getDataFile("step1", "meg_sync"))
  casl_tasks = getCaslTaskFiles() |> getTasks |> Set
  meg_sync_tasks = megsync_dir |> getTaskFiles |> getTasks |> Set
  @assert Set(meg_tasks) == Set(casl_tasks)
  
  casl_tasks, meg_sync_tasks
end
