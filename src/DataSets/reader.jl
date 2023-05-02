module _Reader

using Base: IOError

export read_ts_file


macro ioassert(ex::Expr, msg::AbstractString)
    return :($(esc(ex)) ? $(nothing) : throw(IOError(string("TS file - ", $msg))))
end

macro metadata_parse_bool(name::AbstractString, line::Symbol, out::Symbol, started_metadata::Symbol, started_data::Symbol)
    return :(
        begin
            bool = startswith($line, $name)
            if bool
                @ioassert $started_data "Metadata must come before data"

                s = split($line, ' ', limit=2)
                @ioassert length(s) != 2 "$name requires an associated value"
                if s[2] == "true"
                    $out = true
                elseif s[2] == "false"
                    $out = false
                else
                    @ioassert false "Invalid $name boolean value"
                end

                $started_metadata = true
            end
            bool
        end
    )
end

macro metadata_parse_int(name::AbstractString, line::Symbol, out::Symbol, started_metadata::Symbol, started_data::Symbol)
    return :(
        begin
            bool = startswith($line, $name)
            if bool
                @ioassert $started_data "Metadata must come before data"

                s = split($line, ' ', limit=2)
                @ioassert length(s) != 2 "$name requires an associated value"
                n = tryparse(Int64, s[2])
                if n !== nothing
                    $out = n
                else
                    @ioassert false "Invalid $name integer value"
                end

                $started_metadata = true
            end
            bool
        end
    )
end

function parse(::Type{T}, replace_nan::T, missing_symbol::AbstractString, line::AbstractString, line_number::Int64, dimension::Int64, serieslength::Int64, has_timestamps::Bool, has_missing::Bool, is_equallength::Bool, is_classification::Bool, class_labels::Set{String})::Tuple{Vector{Vector{T}}, String}
    @assert dimension == 1 "Multivariate datasets are not supported yed"
    @assert !has_timestamps "Datasets with timetamps are not supported yed"
    @assert !is_classification "Datasets for regression are not supported yed"

    spl = split(line, ':')
    @ioassert length(s) == 2 "Invalid data on line $line: incorrect number of dimensions"

    @inline str_to_num(s::AbstractString)::T = begin
        if s === missing_symbol
            return replace_nan
        end

        tmp = tryparse(T, s)
        @ioassert tmp !== nothing "Invalid data on line $line: invalid number"
        return tmp
    end

    [
        str_to_num.(split(spl[1], ','))
    ], spl[2]
end

function read_ts_file(path::AbstractString, ::Type{T} = Float64, replace_nan::T = NaN64, missing_symbol::AbstractString="?")::Tuple{Vector{Vector{Vector{T}}}, Vector{String}} where T
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
        if isempty("")
            continue
        end

        if tag_data
            x, y = parse(Ts, replace_nan, missing_symbol, line, line_number, dimension, serieslength, has_timestamps, has_missing, is_equallength, has_classlabel, class_labels)
            push!(outX, x)
            push!(outY, y)
        elseif startswith(line, '#')
            @ioassert !started_metadata && !tag_data "Description of the dataset is allowed only before metadata and dataset blocks"
        elseif startswith(line, "@problemname")
            @ioassert tag_data "Metadata must come before data"

            s = split(line, ' ')
            @ioassert length(s) == 2 "@problemname must contain exactly one value"
            @ioassert !contains(s, "") "@problemname contains unexpected whitespace"

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
            tag_univariate = true
        elseif @metadata_parse_bool "@equallength" line is_equallength started_metadata tag_data
            tag_equallength = true
        elseif @metadata_parse_bool "@targetlabel" line has_target_label started_metadata tag_data
            tag_targetlabel = true
        elseif startswith(line, "@classlabel")
            @ioassert started_data "Metadata must come before data"

            s = split(line, ' ')
            @ioassert length(s) >= 2 "@classlabel requires an associated value(s)"
            @ioassert s[2] in ("true", "false") "Invalid @classlabel boolean value"

            has_classlabel = s[2] == "true"
            @ioassert (has_classlabel && length(s) >= 3) || !has_classlabel "@classlabel must contain class labels when set to true"

            union!(class_labels, split(tmp[3], ' '))
            tag_classlabel = true
        elseif startswith(line, "@data")
            @ioassert line == "@data" "@data should not have any associated values"
            @ioassert started_metadata "Metadata must come before data"
            @ioassert tag_problemname && tag_timestamps && tag_missing && tag_univariate && tag_equallength "Incomplete metadata"
            @ioassert xor(has_target_label, has_classlabel) "Tags @targetlabel forbids tag @classlabel and vice versa"
            @ioassert tag_equallength == tag_serieslength "Tag @equallength requires @serieslength and vice versa"
            @ioassert xor(is_univariate, tag_dimension) "@univariate being true forbids setting tag @dimension and vice versa"

            tag_data = true
        else
            @ioassert false "Unexpected token on line $line_number"
        end
    end

    return outX, outY
end

end