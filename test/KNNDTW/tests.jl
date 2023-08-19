using TsClassification: KNNDTW
using Test: @testset, @test, @test_throws
using CategoricalArrays: CategoricalArray, categorical
using CategoricalDistributions: pdf
using MLJ: machine, fit!, fitted_params, predict, predict_mode


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
    @test sqrt(r1) ≈ 0.8554589614450465

    r2 = KNNDTW.dtw!(model, TS1, TS3)
    @test sqrt(r2) ≈ 1.196169334904773

    r3 = KNNDTW.dtw!(model, TS1, TS1)
    @test sqrt(r3) ≈ 0

    r4 = KNNDTW.dtw!(model, TS1, TS4)
    @test sqrt(r4) ≈ 0.5183078077939663
end

@testset "KNNDTW.jl - dtw!() - SakoeChiba" begin
    model = KNNDTW.DTWSakoeChiba{eltype(TS1)}(radius=2)
    r1 = KNNDTW.dtw!(model, TS1, TS2)
    @test sqrt(r1) ≈ 1.3407807929942596e154

    r2 = KNNDTW.dtw!(model, TS1, TS3)
    @test sqrt(r2) ≈ 1.6133200246629615

    r3 = KNNDTW.dtw!(model, TS1, TS1)
    @test sqrt(r3) ≈ 0

    r4 = KNNDTW.dtw!(model, TS1, TS4)
    @test sqrt(r4) ≈ 0.5183078077939663
end

@testset "KNNDTW.jl - dtw!() - Itakura" begin
    model = KNNDTW.DTWItakura{eltype(TS1)}(slope=1.5)
    r1 = KNNDTW.dtw!(model, TS1, TS2)
    @test sqrt(r1) ≈ 1.0915468341537107

    r2 = KNNDTW.dtw!(model, TS1, TS3)
    @test sqrt(r2) ≈ 1.6803668615702465

    r3 = KNNDTW.dtw!(model, TS1, TS1)
    @test sqrt(r3) ≈ 0

    r4 = KNNDTW.dtw!(model, TS1, TS4)
    @test sqrt(r4) ≈ 0.5183078077939663
end

@testset "KNNDTW.jl - lower_bound!() - LBNone" begin
    lb = KNNDTW.LBNone()
    @test KNNDTW.lower_bound!(lb, TS3, TS4) == 0
end

@testset "KNNDTW.jl - lower_bound!() - LBKeogh" begin
    lb = KNNDTW.LBKeogh{eltype(TS1)}(radius=1)
    @test KNNDTW.lower_bound!(lb, TS3, TS4) ≈ 5.01243886
    @test KNNDTW.lower_bound!(lb, TS1, TS4, update = false) ≈ 5.01243886
    @test KNNDTW.lower_bound!(lb, TS1, TS4) ≈ 0.94498057
    @test_throws AssertionError KNNDTW.lower_bound!(lb, TS1, TS2)

    dtw = KNNDTW.DTWSakoeChiba{eltype(TS1)}(radius=0)
    @test KNNDTW.dtw!(dtw, TS1, TS4) <= KNNDTW.lower_bound!(lb, TS1, TS4)
end

@testset "KNNDTW.jl - KNN - K=1" begin
    nn = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TS1)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict(mach, (TEST_X, :column_based))

    @test pdf.(pred, "a") ≈ [1, 0]
    @test pdf.(pred, "b") ≈ [0, 1]
end

@testset "KNNDTW.jl - KNN - K=3 - Xnew smaller than X" begin
    nn = KNNDTW.KNNDTWModel(K=3, weights=:distance, distance=KNNDTW.DTW{eltype(TS1)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict(mach, (TEST_X, :column_based))

    @test pdf.(pred, "a") ≈ [1, 0.5127566894835097]
    @test pdf.(pred, "b") ≈ [0, 0.4872433105164903]
end

@testset "KNNDTW.jl - KNN - K=3 - Xnew larger than X" begin
    nn = KNNDTW.KNNDTWModel(K=3, weights=:distance, distance=KNNDTW.DTW{eltype(TS1)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict(mach, (hcat(TEST_X, TEST_X), :column_based))

    @test pdf.(pred, "a") ≈ [1, 0.5127566894835097, 1, 0.5127566894835097]
    @test pdf.(pred, "b") ≈ [0, 0.4872433105164903, 0, 0.4872433105164903]
end

@testset "KNNDTW.jl - KNN - K=3 - predict_mode" begin
    nn = KNNDTW.KNNDTWModel(K=3, weights=:distance, distance=KNNDTW.DTW{eltype(TS1)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict_mode(mach, (TEST_X, :column_based))
    @test pred == ["a", "a"]
end

@testset "KNNDTW.jl - KNN - K=1 - different lengths" begin
    nn = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TS1)}())

    mach = machine(nn, (TS1[:,:], :column_based), [1337])
    fit!(mach, verbosity=0)
    pred = predict(mach, (TS2[:,:], :column_based))
    @test pdf.(pred, 1337) ≈ [1.0]
end

@testset "KNNDTW.jl - KNN - K=1 - repeated is same" begin
    nn1 = KNNDTW.KNNDTWModel(K=2, distance=KNNDTW.DTW{eltype(TS1)}())
    nn2 = KNNDTW.KNNDTWModel(K=2, distance=KNNDTW.DTW{eltype(TS1)}())

    mach1 = machine(nn1, (TRAIN_X, :column_based), TRAIN_Y)
    mach2 = machine(nn2, (TRAIN_X, :column_based), TRAIN_Y)

    fit!(mach1, verbosity=0)
    fit!(mach2, verbosity=0)

    pred1 = predict_mode(mach1, (TEST_X, :column_based))
    pred2 = predict_mode(mach2, (TEST_X, :column_based))

    @test pred1 == pred2
end
