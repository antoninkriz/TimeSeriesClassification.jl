using TimeSeriesClassification: MiniRocketModel
using Test: @testset, @test
using Random: Xoshiro
using MLJBase: machine, fit!, fitted_params, transform
import MLJModelInterface

include("consts.jl")


@testset "MiniRocket.jl - fit() can run" begin
    m_machine = MiniRocketModel(num_features=Unsigned(84), rng=Xoshiro(1337))
    m_model = MiniRocketModel(num_features=Unsigned(84), rng=Xoshiro(1337))

    mach = machine(m_machine, transpose(M_FIT1))
    fit!(mach, verbosity=0)

    result_machine = fitted_params(mach)
    result_model = MLJModelInterface.fit(m_model, false, M_FIT1)[1]

    @test result_machine.dilations == result_model[1]
    @test result_machine.num_features_per_dilation == result_model[2]
    @test result_machine.biases ≈ result_model[3]

    @test result_machine.dilations == DILATIONS1
    @test result_machine.num_features_per_dilation == NUM_FEATURES_PER_DILATION1
    @test result_machine.biases ≈ BIASES1
end

@testset "MiniRocket.jl - fit() - random" begin
    m_machine = MiniRocketModel(num_features=Unsigned(190), rng=Xoshiro(1337))
    m_model = MiniRocketModel(num_features=Unsigned(190), rng=Xoshiro(1337))

    mach = machine(m_machine, transpose(M_FIT2))
    fit!(mach, verbosity=0)

    result_machine = fitted_params(mach)
    result_model = MLJModelInterface.fit(m_model, false, M_FIT2)[1]

    @test result_machine.dilations == result_model[1]
    @test result_machine.num_features_per_dilation == result_model[2]
    @test result_machine.biases ≈ result_model[3]

    @test result_machine.dilations == DILATIONS2
    @test result_machine.num_features_per_dilation == NUM_FEATURES_PER_DILATION2
    @test result_machine.biases ≈ BIASES2_RNG
end

@testset "MiniRocket.jl - fit() - shuffled" begin
    m = MiniRocketModel(num_features=Unsigned(190), shuffled=true)
    mach = machine(m, transpose(M_FIT2))
    fit!(mach, verbosity=0)

    result_machine = fitted_params(mach)
    result_model = MLJModelInterface.fit(m, false, M_FIT2)[1]

    @test result_machine.dilations == result_model[1]
    @test result_machine.num_features_per_dilation == result_model[2]
    @test result_machine.biases ≈ result_model[3]

    @test result_machine.dilations == DILATIONS2
    @test result_machine.num_features_per_dilation == NUM_FEATURES_PER_DILATION2
    @test result_machine.biases ≈ BIASES2
end

@testset "MiniRocket.jl - transform() can run" begin
    m_machine = MiniRocketModel(num_features=Unsigned(84), rng=Xoshiro(1337))
    m_model = MiniRocketModel(num_features=Unsigned(84), rng=Xoshiro(1337))

    mach = machine(m_machine, transpose(M_FIT1))
    fit!(mach, verbosity=0)
    t_machine = transform(mach, transpose(TRANSFORM1))

    fp_model =  MLJModelInterface.fit(m_model, false, M_FIT1)[1]
    t_model = MLJModelInterface.transform(m_model, fp_model, TRANSFORM1)

    @test t_machine ≈ t_model
    @test t_machine ≈ transpose(TRANSFORMED1)
end

@testset "MiniRocket.jl - transform() - large" begin
    m_machine = MiniRocketModel(num_features=Unsigned(84 * 4), rng=Xoshiro(1337))

    mach = machine(m_machine, transpose(M_FIT3))
    fit!(mach, verbosity=0)
    t_machine = transform(mach, transpose(TRANSFORM3))

    @test t_machine ≈ transpose(TRANSFORMED3)
end

@testset "MiniRocket.jl - row major and column major inputs" begin
    data_col = M_FIT2
    data_row = permutedims(M_FIT2)

    m_machine_col = MiniRocketModel(num_features=Unsigned(190), shuffled=true)
    m_machine_row = MiniRocketModel(num_features=Unsigned(190), shuffled=true)
    m_model = MiniRocketModel(num_features=Unsigned(190), shuffled=true)

    mach_col = machine(m_machine_col, transpose(data_col))
    mach_row = machine(m_machine_row, data_row)

    fit!(mach_col, verbosity=0)
    fit!(mach_row, verbosity=0)
    fp_model =  MLJModelInterface.fit(m_model, false, M_FIT2)[1]

    res_col_c = transform(mach_col, transpose(data_col))
    res_row_c = transform(mach_row, transpose(data_col))
    res_model = MLJModelInterface.transform(m_model, fp_model, M_FIT2)
    @test res_col_c ≈ res_row_c ≈ res_model

    res_col_r = transform(mach_col, data_row)
    res_row_r = transform(mach_row, data_row)
    @test res_col_r ≈ res_row_r

    @test res_col_c ≈ res_row_c ≈ res_model ≈ res_col_r ≈ res_row_r
end
