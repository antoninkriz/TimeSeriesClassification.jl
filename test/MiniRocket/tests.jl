using TsClassification: MiniRocket
using Test: @testset, @test


const M_FIT::Matrix{Float64} = [
    1 4 7 0  1 9 10
    2 5 8 0 -1 9 11
    3 6 7 0 -1 9 12
    1 4 8 0  1 9 13
    2 5 7 0  1 9 14
    3 6 8 0 -1 9 15
    1 4 7 0 -1 9 16
    2 5 8 0  1 9 17
    3 6 7 0 -1 9 18
    1 4 7 0  1 9 10
    2 5 8 0 -1 9 11
    3 6 7 0 -1 9 12
    1 4 8 0  1 9 13
    2 5 7 0  1 9 14
    3 6 8 0 -1 9 15
    1 4 7 0 -1 9 16
    2 5 8 0  1 9 17
    3 6 7 0 -1 9 18
]
const M_TRANSFORM::Matrix{Float64} = [
    1 0
    2 0
    3 0
    4 0
    5 0
    6 0
    7 0
    8 0
    9 0
    1 0
    2 0
    3 0
    4 0
    5 0
    6 0
    7 0
    8 0
    9 0
]
const DILATIONS = Unsigned[0x0000000000000001, 0x0000000000000002]
const NUM_FEATURES_PER_DILATION = Unsigned[0x0000000000000001, 0x0000000000000001]
const BIASES::Vector{Float64} = [-1.0, -10.0, 11.71706768869104, 0.0, -4.868443825035733, -9.0, -38.17768737133755, 4.263112349928534, -2.44079972126608, -17.539553387625066, 0.0, 0.0, -30.73039637645931, -8.72373203257504, -6.40133286877677, 0.42097975988565395, -8.0, 0.0, 1.0, -9.0, 11.29574178826185, -0.8552882075392461, -10.0, -4.84213259004288, 0.0, -3.8289769725465135, -9.0, -37.07910677525013, 5.07226936109447, -2.2106629502141573, -13.776527572832137, 0.0, -1.2829323113089899, -33.98719052304726, -6.809330081437508, -5.289596655192959, 1.0, -6.500086535135665, 0.0, 4.0, -9.0, 9.5914835765237, -1.217154223826796, -8.710576415078492, -2.2039986063304298, 0.0, -6.57252896650219, -9.0, -35.53306211401237, 6.328890437410607, -0.49359526152326794, -9.657953945093027, 0.0, -1.0, 16.75601533036479, -3.0, -6.625324506758744, 3.0, -7.111909283855383, 0.0, 1.0, -9.0, 9.0, 0.0, -7.072442431366042, 3.8682707547640405, 0.0, -6.5527090051216135, -9.0, -21.618660162875017, 6.0, -5.0, -19.099099806901677, 0.0, -2.0, 22.499221183779014, -4.987017452775206, -5.486930917639299, 0.11788134665258099, -8.0, 0.0, 2.5393803173536753, -9.0, 40.4202874787984, 0.059113743598345536, -24.17154223826796, 18.14453872218894, 0.0, -4.914575021408922, -9.0, -53.60567761565119, 6.605158404835748, -2.553055145664757, -24.67180184367689, 0.0, -1.368530360171519, 9.0, -4.710749485349822, -6.848796933926849, 0.0, -13.013847898581446, 0.0, 1.1775143010658837, -9.0, 14.762679674249966, -3.0, -14.592348927882767, -6.0, 0.0, 5.447117924606573, -18.0, -48.15942504240127, 6.0, 1.7493509864825114, -27.204863957688048, 0.0, -2.0, 6.985632890603114, 0.5655184820745944, -3.631988850642472, 3.59182971706781, -8.0, 0.0, 0.8156482847785753, -9.0, 9.8288039022747, -0.6646182889767545, -11.158040480228692, -1.0, 0.0, 6.723385892031956, -18.0, -54.12575718243707, 6.881426372260648, 0.7760083620174214, -22.0, 0.0, -1.092262392746619, 12.72883874401299, 3.0, -8.145057933003898, 2.9340488422461135, -8.0, 0.0, 1.0, -18.356759304846037, 4.2024409738865245, -4.0, -2.039812993032001, 7.0, 0.0, 7.999653859457339, 0.0, -28.90912216942644, 3.5195603559728568, -7.8693091763929885, -29.467284026530535, 0.0, -2.0, 18.472044597427214, -6.881945583074639, -6.0, 5.572182825958805, -21.763718095877948, 0.0, 1.0, -45.0, 19.4710061757944]
const TRANSFORMED::Vector{Float64} = [-1.0, -10.0, 11.71706768869104, 0.0, -4.868443825035733, -9.0, -38.17768737133755, 4.263112349928534, -2.44079972126608, -17.539553387625066, 0.0, 0.0, -30.73039637645931, -8.72373203257504, -6.40133286877677, 0.42097975988565395, -8.0, 0.0, 1.0, -9.0, 11.29574178826185, -0.8552882075392461, -10.0, -4.84213259004288, 0.0, -3.8289769725465135, -9.0, -37.07910677525013, 5.07226936109447, -2.2106629502141573, -13.776527572832137, 0.0, -1.2829323113089899, -33.98719052304726, -6.809330081437508, -5.289596655192959, 1.0, -6.500086535135665, 0.0, 4.0, -9.0, 9.5914835765237, -1.217154223826796, -8.710576415078492, -2.2039986063304298, 0.0, -6.57252896650219, -9.0, -35.53306211401237, 6.328890437410607, -0.49359526152326794, -9.657953945093027, 0.0, -1.0, 16.75601533036479, -3.0, -6.625324506758744, 3.0, -7.111909283855383, 0.0, 1.0, -9.0, 9.0, 0.0, -7.072442431366042, 3.8682707547640405, 0.0, -6.5527090051216135, -9.0, -21.618660162875017, 6.0, -5.0, -19.099099806901677, 0.0, -2.0, 22.499221183779014, -4.987017452775206, -5.486930917639299, 0.11788134665258099, -8.0, 0.0, 2.5393803173536753, -9.0, 40.4202874787984, 0.059113743598345536, -24.17154223826796, 18.14453872218894, 0.0, -4.914575021408922, -9.0, -53.60567761565119, 6.605158404835748, -2.553055145664757, -24.67180184367689, 0.0, -1.368530360171519, 9.0, -4.710749485349822, -6.848796933926849, 0.0, -13.013847898581446, 0.0, 1.1775143010658837, -9.0, 14.762679674249966, -3.0, -14.592348927882767, -6.0, 0.0, 5.447117924606573, -18.0, -48.15942504240127, 6.0, 1.7493509864825114, -27.204863957688048, 0.0, -2.0, 6.985632890603114, 0.5655184820745944, -3.631988850642472, 3.59182971706781, -8.0, 0.0, 0.8156482847785753, -9.0, 9.8288039022747, -0.6646182889767545, -11.158040480228692, -1.0, 0.0, 6.723385892031956, -18.0, -54.12575718243707, 6.881426372260648, 0.7760083620174214, -22.0, 0.0, -1.092262392746619, 12.72883874401299, 3.0, -8.145057933003898, 2.9340488422461135, -8.0, 0.0, 1.0, -18.356759304846037, 4.2024409738865245, -4.0, -2.039812993032001, 7.0, 0.0, 7.999653859457339, 0.0, -28.90912216942644, 3.5195603559728568, -7.8693091763929885, -29.467284026530535, 0.0, -2.0, 18.472044597427214, -6.881945583074639, -6.0, 5.572182825958805, -21.763718095877948, 0.0, 1.0, -45.0, 19.4710061757944]


@testset "MiniRocket.jl - fit() can run" begin
    d, n, b = MiniRocket.fit(M_FIT, num_features=Unsigned(190))
    @test d == DILATIONS
    @test n == NUM_FEATURES_PER_DILATION
    @test b ≈ BIASES
end


@testset "MiniRocket.jl - transform() can run" begin
    t = MiniRocket.transform(M_TRANSFORM, dilations=DILATIONS, num_features_per_dilation=NUM_FEATURES_PER_DILATION, biases=BIASES)
    @test t ≈ TRANSFORMED
end