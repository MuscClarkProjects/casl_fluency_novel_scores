project_folder = dirname(dirname(pwd()))


getDataFile(parts...) = joinpath(project_folder, "data", parts...)
