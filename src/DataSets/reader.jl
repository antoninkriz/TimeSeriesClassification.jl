module _Reader

using Base: IOError

export read_ts_file


macro read_assert(ex::Union{Symbol, Expr, Bool}, msg::Union{Expr, AbstractString})
    return :($(esc(ex)) ? $(nothing) : throw(AssertionError(string("TS file - ", $msg))))
end


macro metadata_parse_bool(name::AbstractString, line::Symbol, out::Symbol, started_metadata::Symbol, started_data::Symbol)
    return esc(
        quote
            begin
                bool = startswith($line, $name)
                if bool
                    @read_assert !$started_data "Metadata must come before data"

                    s = split($line, ' ', limit=2)
                    @read_assert length(s) == 2 "$($name) requires an associated value"
                    if s[2] == "true"
                        $out = true
                    elseif s[2] == "false"
                        $out = false
                    else
                        @read_assert false "Invalid $($name) boolean value"
                    end

                    $started_metadata = true
                end
                bool
            end
        end
    )
end

macro metadata_parse_int(name::AbstractString, line::Symbol, out::Symbol, started_metadata::Symbol, started_data::Symbol)
    return esc(
        quote
            bool = startswith($line, $name)
            if bool
                @read_assert !$started_data "Metadata must come before data"

                s = split($line, ' ', limit=2)
                @read_assert length(s) == 2 "$($name) requires an associated value"
                n = tryparse(Int64, s[2])
                if n !== nothing
                    $out = n
                else
                    @read_assert false "Invalid $($name) integer value"
                end

                $started_metadata = true
            end
            bool
        end
    )
end

function parse(::Type{T}, replace_missing_by::T, missing_symbol::AbstractString, line::AbstractString, line_number::Int64, dimension::Int64, series_length::Int64, has_timestamps::Bool, has_missing::Bool, is_equallength::Bool, is_classification::Bool, class_labels::Set{String})::Tuple{Vector{Vector{T}}, String} where {T}
    @read_assert dimension == 1 "Multivariate datasets are not supported yed"
    @read_assert !has_timestamps "Datasets with timetamps are not supported yed"
    @read_assert is_classification "Datasets for regression are not supported yed"

    spl = split(line, ':')
    @read_assert length(spl) == 2 "Invalid data on line $line: incorrect number of dimensions"

    @inline str_to_num(s::AbstractString)::T = begin
        if s === missing_symbol
            return replace_missing_by
        end

        tmp = tryparse(T, s)
        @read_assert tmp !== nothing "Invalid data on line $line: invalid number"
        return tmp
    end

    [
        str_to_num.(split(spl[1], ','))
    ], spl[2]
end

function read_ts_file(path::AbstractString, ::Type{T} = Float64, replace_missing_by::T = NaN64, missing_symbol::AbstractString="?")::Tuple{Vector{Vector{Vector{T}}}, Vector{String}} where T
    # Parsing info
    started_metadata::Bool = false

    # Misc info
    problem_name::String = ""
    dimension::Int64 = 0
    series_length::Int64 = 0
    has_timestamps::Bool = false
    has_missing::Bool = false
    is_univariate::Bool = false
    is_equallength::Bool = false
    has_target_label::Bool = false
    has_classlabel::Bool = false
    class_labels::Set{String} = Set{String}()

    # Tags present
    tag_problemname::Bool = false
    tag_dimension::Bool = false
    tag_serieslength::Bool = false
    tag_timestamps::Bool = false
    tag_missing::Bool = false
    tag_univariate::Bool = false
    tag_equallength::Bool = false
    tag_targetlabel::Bool = false
    tag_classlabel::Bool = false
    tag_data::Bool = false

    outX::Vector{Vector{Vector{T}}} = []
    outY::Vector{String} = []

    for (line_number, line) in enumerate(eachline(path))
        line = lowercase(strip(line))

        # Skip blank lines
        if isempty(line)
            continue
        end

        if tag_data
            x, y = parse(T, replace_missing_by, missing_symbol, line, line_number, dimension, series_length, has_timestamps, has_missing, is_equallength, has_classlabel, class_labels)
            push!(outX, x)
            push!(outY, y)
        elseif startswith(line, '#')
            @read_assert !started_metadata && !tag_data "Description of the dataset is allowed only before metadata and dataset blocks"
        elseif startswith(line, "@problemname")
            @read_assert !tag_data "Metadata must come before data"

            s = split(line, ' ')
            @read_assert length(s) == 2 "@problemname must contain exactly one value"
            @read_assert s[2] != "" "@problemname contains unexpected whitespace"

            started_metadata = true
            problem_name = s[2]
            tag_problemname = true
        elseif @metadata_parse_int "@dimension" line dimension started_metadata tag_data
            tag_dimension = true
        elseif @metadata_parse_int "@serieslength" line series_length started_metadata tag_data
            tag_serieslength = true
        elseif @metadata_parse_bool "@timestamps" line has_timestamps started_metadata tag_data
            tag_timestamps = true
        elseif @metadata_parse_bool "@missing" line has_missing started_metadata tag_data
            tag_missing = true
        elseif @metadata_parse_bool "@univariate" line is_univariate started_metadata tag_data
            dimension = 1
            tag_univariate = true
        elseif @metadata_parse_bool "@equallength" line is_equallength started_metadata tag_data
            tag_equallength = true
        elseif @metadata_parse_bool "@targetlabel" line has_target_label started_metadata tag_data
            tag_targetlabel = true
        elseif startswith(line, "@classlabel")
            @read_assert !tag_data "Metadata must come before data"

            s = split(line, ' ', limit=3)
            @read_assert length(s) >= 2 "@classlabel requires an associated value(s)"
            @read_assert s[2] in ("true", "false") "Invalid @classlabel boolean value"

            has_classlabel = s[2] == "true"
            @read_assert (has_classlabel && length(s) >= 3) || !has_classlabel "@classlabel must contain class labels when set to true"

            union!(class_labels, split(s[3], ' '))
            tag_classlabel = true
            started_metadata = true
        elseif startswith(line, "@data")
            @read_assert line == "@data" "@data should not have any associated values"
            @read_assert started_metadata "Metadata must come before data"
            @read_assert tag_problemname && tag_timestamps && tag_missing && tag_univariate && tag_equallength "Incomplete metadata"
            @read_assert xor(has_target_label, has_classlabel) "Tags @targetlabel forbids tag @classlabel and vice versa"
            @read_assert tag_equallength == tag_serieslength "Tag @equallength requires @serieslength and vice versa"
            @read_assert xor(is_univariate, tag_dimension) "@univariate being true forbids setting tag @dimension and vice versa"

            tag_data = true
        else
            @read_assert false "Unexpected token on line $line_number"
        end
    end

    return outX, outY
end

end