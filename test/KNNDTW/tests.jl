using TsClassification: KNNDTW
using Test: @testset, @test
using CategoricalArrays: CategoricalArray, categorical
using CategoricalDistributions: pdf
using MLJ: machine, fit!, fitted_params, predict


const TS1::Vector{Float64} = [0.57173714, 0.03585991, 0.16263380, 0.63153396, 0.00599358, 0.63256182, 0.85341386, 0.87538411, 0.35243848, 0.27466851]
const TS2::Vector{Float64} = [0.17281271, 0.54244937, 0.35081248, 0.83846642, 0.74942411]
const TS3::Vector{Float64} = [1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 0.16263380, 0.63153396, 0.00599358, 0.63256182]
const TS4::Vector{Float64} = [0.57173714, 0.03585991, 0.16263380, 0.00599358, 0.00599358, 0.00599358, 0.85341386, 0.87538411, 0.35243848, 0.27466851]

const TRAIN_X::Matrix{Float64} = [
    0 1 3
    0 2 0
    0 1 3
    1 0 0
    2 0 3
    1 0 0
    0 0 3
]
const TRAIN_Y::CategoricalArray = categorical(["a", "a", "b"])
const TEST_X::Matrix{Float64} = [
    0 8
    0 1
    1 8
    2 1
    1 8
    0 1
    0 8
]

@testset "KNNDTW.jl - dtw!() - Full" begin
    model = KNNDTW.DTW{eltype(TS1)}()
    r1 = KNNDTW.dtw!(model, TS1, TS2)
    @test r1 ≈ 0.8554589614450465

    r2 = KNNDTW.dtw!(model, TS1, TS3)
    @test r2 ≈ 1.196169334904773

    r3 = KNNDTW.dtw!(model, TS1, TS1)
    @test r3 ≈ 0

    r4 = KNNDTW.dtw!(model, TS1, TS4)
    @test r4 ≈ 0.5183078077939663
end

@testset "KNNDTW.jl - dtw!() - SakoeChiba" begin
    model = KNNDTW.DTWSakoeChiba{eltype(TS1)}(radius=2)
    r1 = KNNDTW.dtw!(model, TS1, TS2)
    @test r1 ≈ 1.3407807929942596e154

    r2 = KNNDTW.dtw!(model, TS1, TS3)
    @test r2 ≈ 1.6133200246629615

    r3 = KNNDTW.dtw!(model, TS1, TS1)
    @test r3 ≈ 0

    r4 = KNNDTW.dtw!(model, TS1, TS4)
    @test r4 ≈ 0.5183078077939663
end

@testset "KNNDTW.jl - dtw!() - Itakura" begin
    model = KNNDTW.DTWItakura{eltype(TS1)}(slope=1.5)
    r1 = KNNDTW.dtw!(model, TS1, TS2)
    @test r1 ≈ 1.0915468341537107

    r2 = KNNDTW.dtw!(model, TS1, TS3)
    @test r2 ≈ 1.6803668615702465

    r3 = KNNDTW.dtw!(model, TS1, TS1)
    @test r3 ≈ 0

    r4 = KNNDTW.dtw!(model, TS1, TS4)
    @test r4 ≈ 0.5183078077939663
end

@testset "KNNDTW.jl - KNN - K=1" begin
    nn = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TS1)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict(mach, (TEST_X, :column_based))

    @test pdf.(pred, "a") == [1, 0]
    @test pdf.(pred, "b") == [0, 1]
end

@testset "KNNDTW.jl - KNN - K=3" begin
    nn = KNNDTW.KNNDTWModel(K=3, weights=:distance, distance=KNNDTW.DTW{eltype(TS1)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y, [1.0, 2.0, 3.5])
    fit!(mach, verbosity=0)
    pred = predict(mach, (TEST_X, :column_based))

    @test pdf.(pred, "a") == [1, 0.2966868988076252]
    @test pdf.(pred, "b") == [0, 0.7033131011923748]
end
