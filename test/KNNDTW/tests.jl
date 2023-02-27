using TsClassification: KNNDTW
using Test: @testset, @test


const TS1::Vector{Float64} = [0.57173714, 0.03585991, 0.1626338 , 0.63153396, 0.00599358, 0.63256182, 0.85341386, 0.87538411, 0.35243848, 0.27466851]
const TS2::Vector{Float64} = [0.17281271, 0.54244937, 0.35081248, 0.83846642, 0.74942411]


@testset "KNNDTW.jl - dtw() base" begin
    r = KNNDTW.dtw(TS1, TS2)
    @test r ≈ 0.8554589614450465
end

@testset "KNNDTW.jl - dtw() sakoe_chiba_radius" begin
    r = KNNDTW.dtw_with_sakoe_chiba_radius(TS1, TS2, sakoe_chiba_radius=Unsigned(2))
    @test r === nothing
end

@testset "KNNDTW.jl - dtw() itakura_max_slope" begin
    r = KNNDTW.dtw_with_itakura_max_slope(TS1, TS2, itakura_max_slope=2)
    @test r === nothing
end
