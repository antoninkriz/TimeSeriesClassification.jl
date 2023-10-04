using TimeSeriesClassification: KNNDTW
using Test: @testset, @test, @test_throws
using CategoricalArrays: CategoricalArray, categorical
using CategoricalDistributions: pdf
using MLJBase: machine, fit!, fitted_params, predict, predict_mode
import MLJModelInterface

include("consts.jl")


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
    nn = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TRAIN_X)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict(mach, (TEST_X, :column_based))

    @test pdf.(pred, "a") ≈ [1, 0]
    @test pdf.(pred, "b") ≈ [0, 1]
end

@testset "KNNDTW.jl - KNN - K=3 - Xnew smaller than X" begin
    nn = KNNDTW.KNNDTWModel(K=3, weights=:distance, distance=KNNDTW.DTW{eltype(TRAIN_X)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict(mach, (TEST_X, :column_based))

    @test pdf.(pred, "a") ≈ [1, 0.5127566894835097]
    @test pdf.(pred, "b") ≈ [0, 0.4872433105164903]
end

@testset "KNNDTW.jl - KNN - K=3 - Xnew larger than X" begin
    nn = KNNDTW.KNNDTWModel(K=3, weights=:distance, distance=KNNDTW.DTW{eltype(TRAIN_X)}())

    mach = machine(nn, (TRAIN_X, :column_based), TRAIN_Y)
    fit!(mach, verbosity=0)
    pred = predict(mach, (hcat(TEST_X, TEST_X), :column_based))

    @test pdf.(pred, "a") ≈ [1, 0.5127566894835097, 1, 0.5127566894835097]
    @test pdf.(pred, "b") ≈ [0, 0.4872433105164903, 0, 0.4872433105164903]
end

@testset "KNNDTW.jl - KNN - K=3 - predict_mode" begin
    nn = KNNDTW.KNNDTWModel(K=3, weights=:distance, distance=KNNDTW.DTW{eltype(TRAIN_X)}())

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
    nn1 = KNNDTW.KNNDTWModel(K=2, distance=KNNDTW.DTW{eltype(TRAIN_X)}())
    nn2 = KNNDTW.KNNDTWModel(K=2, distance=KNNDTW.DTW{eltype(TRAIN_X)}())

    mach1 = machine(nn1, (TRAIN_X, :column_based), TRAIN_Y)
    mach2 = machine(nn2, (TRAIN_X, :column_based), TRAIN_Y)

    fit!(mach1, verbosity=0)
    fit!(mach2, verbosity=0)

    pred1 = predict_mode(mach1, (TEST_X, :column_based))
    pred2 = predict_mode(mach2, (TEST_X, :column_based))

    @test pred1 == pred2
end

@testset "KNNDTW.jl - row major and column major inputs" begin
    data_col = TRAIN_X
    data_row = permutedims(TRAIN_X)
    data_vec = [view(TRAIN_X, :, col) for col in axes(TRAIN_X, 2)]

    m_machine_col = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TRAIN_X)}())
    m_machine_row1 = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TRAIN_X)}())
    m_machine_row2 = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TRAIN_X)}())
    m_machine_vec = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TRAIN_X)}())
    m_model = KNNDTW.KNNDTWModel(K=1, distance=KNNDTW.DTW{eltype(TRAIN_X)}())

    mach_col = machine(m_machine_col, (data_col, :column_based), TRAIN_Y)
    mach_row1 = machine(m_machine_row1, (data_row, :row_based), TRAIN_Y)
    mach_row2 = machine(m_machine_row2, data_row, TRAIN_Y)
    mach_vec = machine(m_machine_vec, data_vec, TRAIN_Y)

    fit!(mach_col, verbosity=0)
    fit!(mach_row1, verbosity=0)
    fit!(mach_row2, verbosity=0)
    fit!(mach_vec, verbosity=0)
    fp_model =  MLJModelInterface.fit(m_model, false, data_vec, TRAIN_Y)[1]

    pred_col_c = predict_mode(mach_col, (data_col, :column_based))
    pred_col_r1 = predict_mode(mach_col, (data_row, :row_based))
    pred_col_r2 = predict_mode(mach_col, data_row)
    pred_col_v = predict_mode(mach_col, data_vec)
    @test pred_col_c == pred_col_r1 == pred_col_r2 == pred_col_v

    pred_row1_c = predict_mode(mach_row1, (data_col, :column_based))
    pred_row1_r1 = predict_mode(mach_row1, (data_row, :row_based))
    pred_row1_r2 = predict_mode(mach_row1, data_row)
    pred_row1_v = predict_mode(mach_row1, data_vec)
    @test pred_row1_c == pred_row1_r1 == pred_row1_r2 == pred_row1_v

    pred_row2_c = predict_mode(mach_row2, (data_col, :column_based))
    pred_row2_r1 = predict_mode(mach_row2, (data_row, :row_based))
    pred_row2_r2 = predict_mode(mach_row2, data_row)
    pred_row2_v = predict_mode(mach_row2, data_vec)
    @test pred_row2_c == pred_row2_r1 == pred_row2_r2 == pred_row2_v

    pred_vec_c = predict_mode(mach_vec, (data_col, :column_based))
    pred_vec_r1 = predict_mode(mach_vec, (data_row, :row_based))
    pred_vec_r2 = predict_mode(mach_vec, data_row)
    pred_vec_v = predict_mode(mach_vec, data_vec)
    @test pred_vec_c == pred_vec_r1 == pred_vec_r2 == pred_vec_v

    @test pred_col_c == pred_col_r1 == pred_col_r2 == pred_col_v == pred_row1_c == pred_row1_r1 == pred_row1_r2 == pred_row1_v == pred_row2_c == pred_row2_r1 == pred_row2_r2 == pred_row2_v == pred_vec_c == pred_vec_r1 == pred_vec_r2 == pred_vec_v
end
