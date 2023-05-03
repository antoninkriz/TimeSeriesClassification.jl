module _Utils

using ProgressMeter: Progress, next!

import ZipFile

export unzip


# Based on https://discourse.julialang.org/t/how-to-extract-a-file-in-a-zip-archive-without-using-os-specific-tools/34585/5
function unzip(file::AbstractString, output_directory::AbstractString="")
    file_full_path = isabspath(file) ? file : joinpath(pwd(), file)
    file_base_path = dirname(file_full_path)

    output_path = (output_directory == ""
        ? file_base_path
        : (
            isabspath(output_directory)
            ? output_directory
            : joinpath(pwd(), output_directory)
    ))

    isdir(output_path) ? nothing : mkpath(output_path)


    archive = ZipFile.Reader(file_full_path)
    p = Progress(length(archive.files), "Extracting - $(length(archive.files)) items:")

    Threads.@threads for f in archive.files
        fullFilePath = joinpath(output_directory, f.name)
        if (endswith(f.name, "/") || endswith(f.name, "\\"))
            mkpath(fullFilePath)
        else
            write(fullFilePath, read(f))
        end

        next!(p)
    end
    close(archive)
end

end
