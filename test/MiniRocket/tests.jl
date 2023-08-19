using TSC: MiniRocketModel
using Test: @testset, @test
using Random: Xoshiro
using MLJ: machine, fit!, fitted_params, transform
import MLJModelInterface

include("consts.jl")


@testset "MiniRocket.jl - fit() can run" begin
    m_machine = MiniRocketModel(num_features=Unsigned(84), rng=Xoshiro(1337))
    m_model = MiniRocketModel(num_features=Unsigned(84), rng=Xoshiro(1337))

    mach = machine(m_machine, (M_FIT1, :column_based))
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

    mach = machine(m_machine, (M_FIT2, :column_based))
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
    mach = machine(m, (M_FIT2, :column_based))
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

    mach = machine(m_machine, (M_FIT1, :column_based))
    fit!(mach, verbosity=0)
    t_machine = transform(mach, (TRANSFORM1, :column_based))

    fp_model =  MLJModelInterface.fit(m_model, false, M_FIT1)[1]
    t_model = MLJModelInterface.transform(m_model, fp_model, TRANSFORM1)

    @test t_machine ≈ t_model
    @test t_machine ≈ TRANSFORMED1
end

@testset "MiniRocket.jl - transform() - large" begin
    m_machine = MiniRocketModel(num_features=Unsigned(84 * 4), rng=Xoshiro(1337))

    mach = machine(m_machine, (M_FIT3, :column_based))
    fit!(mach, verbosity=0)
    t_machine = transform(mach, (TRANSFORM3, :column_based))

    @test t_machine ≈ TRANSFORMED3
end
