const M_FIT1::Matrix{Float64} = [
    1
    2
    3
    4
    5
    6
    7
    8
    9;;
]
const DILATIONS1 = [1]
const NUM_FEATURES_PER_DILATION1 = [1]
const BIASES1::Vector{Float64} = [-15.111456180001682, -22.222912360003363, 9.993788759969732, -9.66873708001009, -15.0, -4.003105620015134, -10.170289890017656, 19.20427863991256, -10.006211240030268, -14.114561800016816, -0.8390269700277031, -8.337474160020179, -23.795721360087555, -2.58212586012948, -9.835921350012597, 4.86680447989238, -3.9473775300143075, -15.024844960121072, 6.646997739904236, -3.1145618000168156, 7.0, 1.5479640399630625, -3.0, 3.0, 1.8203932499368989, -23.14251238015322, 6.990683139954626, -16.043478680211877, 19.99051006968284, -9.702898900176365, -12.0, 1.6501033599192851, -6.0, -22.15804048022892, -1.455314650323544, -9.0, 6.814182009906631, -3.0, -13.38716408026221, 8.625258399798213, -3.0, 10.0, 3.0, -0.45203596003693747, 3.0, 3.873015719922705, -21.334541610276915, 9.0, -14.5760876903704, 26.776741499453124, -4.421325900428656, -5.795721360087555, 0.4641122995542446, 0.0, 29.219460599444574, 2.637680879858749, 0.0, 6.0, 2.712042689950408, -6.780539400554289, 7.201173019897396, 3.2691505196874004, 12.0, 12.0, -2.244651700109216, 13.96583817983344, 5.266217969943682, -9.316080960457839, 6.464285369825916, 3.2971010998235215, 12.0, 8.962732559818392, -11.47705399042934, 16.752242679875508, 7.6407864998739115, 19.646651599361576, 12.0, 1.1449257190822664, 17.597480889933536, 8.584539199058327, -23.821431671568007, 18.860593239862283, -6.006903521116328, 27.0]
const TRANSFORM1::Matrix{Float64} = [
    1 -√1
    2  √2
    1  √3
   -1 -√4
    1  √5
    3  √6
    1 -√7
   -4  √8
    1  √9
]
const TRANSFORMED1::Matrix{Float64} = [
    1.0                1.0
    1.0                1.0
    0.2222222222222222 0.1111111111111111
    1.0                1.0
    1.0                1.0
    0.0                1.0
    0.8888888888888888 1.0
    0.0                0.0
    1.0                1.0
    1.0                1.0
    0.5555555555555556 0.4444444444444444
    0.0                1.0
    1.0                1.0
    1.0                0.0
    0.8888888888888888 0.8888888888888888
    0.0                0.0
    0.6666666666666666 0.5555555555555556
    1.0                1.0
    0.2222222222222222 0.1111111111111111
    1.0                0.0
    0.1111111111111111 0.3333333333333333
    1.0                1.0
    0.5555555555555556 0.6666666666666666
    0.0                1.0
    0.4444444444444444 0.5555555555555556
    1.0                1.0
    0.0                0.2222222222222222
    1.0                1.0
    0.0                0.0
    1.0                1.0
    0.8888888888888888 0.8888888888888888
    1.0                0.0
    0.7777777777777778 0.5555555555555556
    1.0                1.0
    0.7777777777777778 0.4444444444444444
    1.0                1.0
    0.2222222222222222 0.3333333333333333
    0.0                1.0
    1.0                1.0
    1.0                1.0
    0.6666666666666666 0.5555555555555556
    0.0                1.0
    0.4444444444444444 0.5555555555555556
    1.0                0.0
    0.4444444444444444 0.3333333333333333
    1.0                1.0
    1.0                1.0
    0.0                0.0
    1.0                1.0
    0.0                0.0
    0.8888888888888888 0.6666666666666666
    1.0                0.0
    0.4444444444444444 0.3333333333333333
    0.0                1.0
    0.0                0.0
    1.0                0.0
    0.6666666666666666 0.5555555555555556
    0.0                1.0
    0.4444444444444444 0.4444444444444444
    1.0                1.0
    0.2222222222222222 0.6666666666666666
    0.0                0.0
    0.0                0.1111111111111111
    0.0                1.0
    0.7777777777777778 0.8888888888888888
    0.0                0.0
    0.3333333333333333 0.3333333333333333
    1.0                1.0
    0.1111111111111111 0.3333333333333333
    0.0                0.0
    0.0                0.1111111111111111
    0.0                0.0
    0.7777777777777778 0.7777777777777778
    0.0                0.0
    0.2222222222222222 0.1111111111111111
    0.0                0.0
    0.0                0.1111111111111111
    0.0                0.0
    0.0                0.0
    0.0                1.0
    1.0                1.0
    0.0                0.0
    0.7777777777777778 0.6666666666666666
    0.0                0.0
]

const M_FIT2::Matrix{Float64} = [
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
const DILATIONS2 = [1, 2]
const NUM_FEATURES_PER_DILATION2 = [1, 1]
const BIASES2::Vector{Float64} = [0.0, -6.0, 11.5986671312232, 0.0, -2.9342219125178666, 0.0, -22.085598048862558, 6.15786740995712, 0.0, -6.8026657375536, 0.0, 0.07893370497856012, -21.73039637645931, 3.092089322474987, -3.0, 9.0, -1.0, 0.0, 4.0, 0.0, 16.0, 0.0, -3.0, 5.0, 0.0, -5.8289769725465135, 0.0, -16.342392195450238, 5.07226936109447, 3.0, -2.776527572832137, 0.0, -2.0, -24.98719052304726, 6.0, 0.0, 7.0, -1.0, 0.0, 3.2631123499285337, 0.0, 19.0, 3.0, 0.0, 3.0, 0.0, -3.9542149441703174, 0.0, -16.5991863420371, 5.328890437410607, 3.0, -1.0, 0.0, -2.0, 25.75601533036479, 9.0, -3.3751947040552466, 6.0, -0.11190928385538257, 0.0, 4.0, 0.0, 20.74320585341205, -1.7370607203430382, 0.0, 9.0, 0.0, -8.0, 0.0, -4.85598048862505, 3.0, 3.0, -13.03963992276067, 0.0, -2.0, 31.499221183779014, 12.019473820837192, -3.0, 6.0982344555438175, -1.0, 0.0, 2.5393803173536753, 0.0, 47.8677515439486, 0.059113743598345536, -20.17154223826796, 18.14453872218894, 0.0, -5.914575021408922, 0.0, -37.802838807825594, 6.605158404835748, 0.0, -18.908429888340635, 0.0, -0.3685303601715191, 9.0, 1.7355030879010656, -4.0, 7.262246998569708, -5.178206582151205, 0.0, 1.0, 0.0, 14.762679674249966, 0.0, -8.0, 2.0, 0.0, 6.7235589623032865, -9.0, -37.05314168080042, 6.0, 1.7498701972965023, -20.204863957688048, 0.0, -2.0, 9.0, 12.282759241037297, -1.4213259004283145, 7.0, -1.0, 0.0, 1.0, 0.0, 18.0, 0.0, -7.158040480228692, 6.34853732851937, 0.0, 5.361692946015978, -1.1855632071236357, -39.12575718243707, 5.881426372260648, 3.0, -14.105418010243227, 0.0, -2.0, 21.72883874401299, 12.0, -3.572528966501949, 4.868097684492227, 2.0, 0.0, 2.4537822684907837, -9.356759304846037, 15.0, -1.052968610529092, 1.4402805104519985, 11.960013936696186, 0.0, 3.9996538594573394, 9.0, -17.7663141499537, 4.55868106791857, 0.05227632944280458, -22.0, 0.0, -2.0, 27.472044597427214, 13.677081625388041, -5.0, 10.572182825958805, -13.842478730585299, 0.0, 1.0, -30.613553451431812, 40.576078045494]
const BIASES2_RNG::Vector{Float64} = [0.0, 0.0, 11.5986671312232, 0.0, 0.0, 0.0, -3.7237320325750396, 27.47360222987136, 0.6776008362017603, -6.8026657375536, 3.434135377382262, 0.0, -21.73039637645931, 3.0, -3.0, 0.0, 0.0, -4.6447983275964795, 0.0, 0.0, 9.0, 0.0, -1.0, 0.0, -1.0066643438840899, -6.0, 6.0, -5.894928130300158, 19.763025814792627, 2.0, -3.0, 0.0, -1.1317292452359595, -8.0, 2.0, 0.0, 4.0, -0.7500432675678326, -7.243465458819649, 3.2631123499285337, 0.0, 19.0, 3.0, 0.0, 3.7960013936695702, -3.0, -15.099272877174698, 2.0, -2.0, 4.0, 3.0, -1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, -0.11190928385538257, -5.815994425321236, 4.0, -5.329582718496411, 7.6576078045494, 6.0, -1.0, 9.0, 6.0, -11.421672040972908, 6.0, 0.0, 0.0, 3.0, -5.0, 0.0, -2.0, 16.499740394593005, -1.9870174527752056, -2.0, 6.0982344555438175, 0.0, 0.0, 3.0, -3.0, 4.0, 3.118227487196691, -55.17154223826796, 36.43361616656682, -18.0, -10.829150042817844, -6.0, -37.802838807825594, 23.0, 3.0, -13.14505793300438, 3.0, 0.0, 7.41414234572963, 0.6446252573250888, -4.5463908017805466, 7.262246998569708, -15.0, -18.0, 7.3550286021317675, -2.315907890186054, 7.0, -6.211009090757784, -23.592348927882767, 0.0, -5.783018846444776, 28.894235849213146, 0.23013677105183206, -2.0, 27.0, 3.4997403945930046, -20.204863957688048, 4.763025814792627, -21.0, 0.0, 12.282759241037297, -9.210662950214157, 9.0, 0.0, -18.0, 9.63129656955715, 0.0, 7.6576078045494, -1.0, -3.316080960457384, 0.0, -2.0, 9.0, -3.0, 0.0, 0.0, -1.6119958189912893, -21.73792607170259, 1.4011597985053186, -17.09226239274662, 37.82863083200289, 9.0, -3.572528966501949, 4.868097684492227, 0.0, 0.0, 0.0, -18.396399227606707, 15.0, -1.052968610529092, 1.4004675174199974, 3.986671312232062, -1.0, 3.9996538594573394, -1.0, -8.88315707497685, 4.55868106791857, 2.0522763294428046, -47.80370415918321, 18.11788134665258, 3.7293579548279467, 23.262246998570674, 3.0, -14.93439498278974, 4.572182825958805, -2.0, 36.0, 0.0, -15.803011878095958, 2.1050718696996]

const M_FIT3::Matrix{Float32} = [-1.64895104, 1.26268043, -0.38997852, 0.43935018, -1.30580377, 0.18567244, -0.11204975, 0.90447119, 1.25310204, -0.58135326, -0.73039275, -1.54964319, -1.61576733, 0.91488709, 1.64514032, 0.56433642, 0.7371386 , 0.44237929, 0.0264341 , 0.89433812, -1.79962867, 0.90379468, 1.20900113, 1.0497391 , 0.00738636, -1.85054894, -2.14106818, 0.99732979, -0.55911057, -0.45374818, 0.38676842, -0.97747618, -1.57740087, -0.49344244, 1.34856001, 0.65029681, -0.61711938, 0.25022716, 0.79860175, -0.39606311, 0.23652612, 0.5663007 , 1.11797973, -0.53553783, 0.47507943, 1.46368345, -0.8117524 , 0.68606977, -1.03782575, -0.13586365, -0.93036769, -1.19274222, -0.76614435, 0.88389602, 1.40403085, 1.66795953, 0.85141966, -1.38033759, 0.52114054, -1.61299857, -1.62288307, 0.10285921, -1.8998723 , -1.43248201, -0.18010663, 0.41153247, 1.1493737 , 0.51904373, 1.62460004, 0.47973552, 1.3323755 , -0.07441436, -0.48736287, -2.15577147, -0.75463747, 0.42174665, -0.04297407, -1.7065197 , 0.33170522, 1.67087539, -0.53407912, -0.12375105, -1.25725711, 0.22053781, -0.25763797, -0.09526332, -1.33653013, 0.55862091, -0.8021711 , -0.12186213, -0.36997735, 0.13639645, -1.7874745 , -0.17779641, 0.40936507, 0.0944892 , -1.26866816, 0.20342425, 0.84767843, -0.01194966, 0.11532169, 0.84128852, 1.50525583, -0.1069634 , -0.08278655, 0.88469716, -0.15859159, -0.10774266, -0.18737832, -1.40555573, -0.98705749, -0.42233331, -2.11677754, 1.62157265, -0.53761374, 0.03700637, 0.84622612, 1.41190328, 0.14691402, -0.57701766, 0.0672095 , 0.96821651, -0.73682141, -0.02017634, 1.30683648, 1.01070243, 1.12347509][:,:]
const TRANSFORM3::Matrix{Float32} = [-9.78338259e-01, -4.00692403e-01, 2.24768051e+00, 1.02512513e+00, 4.03462233e-01, 3.55965252e-02, -1.81793009e+00, -3.64959450e-01, 7.24468618e-01, 5.52221869e-01, -2.03395324e-01, 7.25521480e-02, -3.18151356e-01, -4.66629138e-01, 8.24520724e-01, 2.09808160e+00, -1.58050621e-01, -2.14700488e-01, 1.10506901e+00, 3.43461741e-01, 2.67789429e-01, 1.40955270e-01, -5.13090247e-01, 6.42636311e-01, 3.25493975e-01, 2.08608796e+00, 3.65570374e-01, 4.03110973e-01, 1.01533608e+00, -1.91115732e+00, -2.57412534e+00, -6.19453420e-02, 2.32955145e+00, 2.17407178e-01, -4.83040711e-01, 7.21357703e-01, 1.21785997e+00, 5.53252254e-01, 5.32502847e-02, 2.42901348e+00, -6.81730504e-01, 2.73691242e+00, 1.12105175e-01, -1.46186173e+00, -2.31276694e-01, 1.49597723e+00, 2.59480634e-01, -2.70625842e-02, 2.90117925e-01, 2.63442853e-01, -7.00088787e-01, 1.48094863e-01, -2.49164902e-01, 1.68390056e-01, 7.09814180e-01, -5.54723350e-01, 1.98101405e-01, 1.38765468e+00, 1.47685825e+00, 1.22002806e+00, -6.41906601e-01, 1.05549996e+00, -9.17719696e-01, -1.65821553e+00, 1.10356209e+00, 1.82510778e+00, 9.16093985e-01, 2.86622264e-02, -2.73329508e-01, 1.26397806e+00, -1.00201186e+00, 2.54644625e+00, 3.47602628e-01, 2.02029649e-03, 1.05122057e+00, 8.79062461e-01, -2.20012978e+00, -1.17249304e+00, -1.64948262e+00, 1.24952100e+00, 4.87338884e-01, -9.80270037e-01, 6.13752285e-01, -5.04728488e-01, 1.75592792e+00, 1.31934129e+00, 1.96953045e+00, 1.06573930e+00, 8.21492471e-02, 2.23281408e-01, -2.27838236e+00, -3.13441001e-02, 5.67689791e-01, -4.29944805e-01, -1.05669231e-01, -6.03719657e-01, 1.65161546e-01, -1.21084667e+00, -1.77487953e+00, -2.77877907e-01, -1.61597097e+00, -1.89741026e+00, -8.89665237e-01, -2.03495605e+00, -1.76806592e+00, -4.93367982e-01, 4.17213107e-01, -7.17563480e-02, 3.19355438e-01, -1.67864816e+00, -1.48562859e+00, -6.04259561e-01, 5.02074985e-01, -9.50649405e-01, 1.74324556e+00, -9.91563867e-01, 8.79441926e-01, -1.50368440e+00, 6.28846354e-01, -3.37617555e-01, -1.31140595e+00, -3.84185741e-01, 3.79595798e-01, -1.48282362e-01, 2.01497009e+00, 1.05217226e+00, 5.46513538e-01][:,:]
const TRANSFORMED3::Matrix{Float32} = [0.39370078, 0.7647059 , 0.16535433, 0.56302524, 0.88188976, 0.37815127, 0.68503934, 0.05882353, 0.4015748 , 0.7815126 , 0.22047244, 0.5714286 , 0.96062994, 0.3529412 , 0.70866144, 0.22689076, 0.48818898, 0.8487395 , 0.27559054, 0.57983196, 0.03149606, 0.40336135, 0.68503934, 0.22689076, 0.56692916, 0.90756303, 0.28346458, 0.71428573, 0.11811024, 0.42857143, 0.8582677 , 0.26050422, 0.61417323, 0.99159664, 0.39370078, 0.73109245, 0.11023622, 0.5294118 , 0.8346457 , 0.26890758, 0.62204725, 0.07563026, 0.43307087, 0.77310926, 0.16535433, 0.5882353 , 0.9055118 , 0.31932774, 0.68503934, 0.11764706, 0.4566929 , 0.8235294 , 0.23622048, 0.64705884, 0.02362205, 0.3529412 , 0.7401575 , 0.22689076, 0.5826772 , 0.84033614, 0.2913386 , 0.68907565, 0.07874016, 0.42016807, 0.79527557, 0.24369748, 0.6614173 , 0.9579832 , 0.37007874, 0.7226891 , 0.16535433, 0.48739496, 0.8267717 , 0.25210086, 0.62992126, 0.05042017, 0.511811 , 0.77310926, 0.20472442, 0.62184876, 0.92913383, 0.3277311 , 0.7322835 , 0.06722689, 0.4144144 , 0.79527557, 0.2972973 , 0.5590551 , 0.981982 , 0.37795275, 0.8018018 , 0.1496063 , 0.5585586 , 0.9055118 , 0.3243243 , 0.7007874 , 0.04504504, 0.43307087, 0.7477477 , 0.2992126 , 0.6126126 , 0.93700784, 0.3963964 , 0.77952754, 0.14414415, 0.503937 , 0.8558559 , 0.26771653, 0.5675676 , 0.02362205, 0.44144145, 0.8110236 , 0.17117117, 0.48031497, 0.8738739 , 0.30708662, 0.7477477 , 0.10236221, 0.5585586 , 0.8582677 , 0.21621622, 0.52755904, 0.954955 , 0.37007874, 0.6936937 , 0.22047244, 0.5945946 , 0.8425197 , 0.2972973 , 0.6614173 , 0.06306306, 0.4566929 , 0.7927928 , 0.21259843, 0.5315315 , 0.92913383, 0.3243243 , 0.72440946, 0.0990991 , 0.54330707, 0.7927928 , 0.24409449, 0.5675676 , 0.00787402, 0.43243244, 0.71653545, 0.13513513, 0.53543305, 0.9279279 , 0.26771653, 0.6666667 , 0.04724409, 0.4144144 , 0.8503937 , 0.2072072 , 0.52755904, 0.9189189 , 0.40944883, 0.6756757 , 0.12598425, 0.4774775 , 0.8503937 , 0.2972973 , 0.5826772 , 0.05405406, 0.36220473, 0.7567568 , 0.17322835, 0.53543305, 0.89873415, 0.36220473, 0.70886075, 0.16535433, 0.39240506, 0.79527557, 0.35443038, 0.6456693 , 0.9493671 , 0.36220473, 0.7468355 , 0.15748031, 0.5949367 , 0.8897638 , 0.30379745, 0.71653545, 0.10126583, 0.41732284, 0.7848101 , 0.21259843, 0.5822785 , 0.97637796, 0.29113925, 0.6692913 , 0.1392405 , 0.496063 , 0.7468355 , 0.2992126 , 0.65822786, 0.01574803, 0.39240506, 0.77165353, 0.20253165, 0.47244096, 0.9113924 , 0.2992126 , 0.6835443 , 0.14173229, 0.443038 , 0.8503937 , 0.21518987, 0.6614173 , 0.9746835 , 0.39370078, 0.6708861 , 0.17322835, 0.49367088, 0.81889766, 0.39240506, 0.6535433 , 0.08860759, 0.4015748 , 0.7468355 , 0.23622048, 0.56962025, 0.9055118 , 0.36708862, 0.6692913 , 0.16455697, 0.40944883, 0.70886075, 0.21259843, 0.556962 , 0.992126 , 0.46835443, 0.77165353, 0.2278481 , 0.53543305, 0.9113924 , 0.2992126 , 0.5949367 , 0.10236221, 0.32911393, 0.7480315 , 0.18987341, 0.56692916, 0.87341774, 0.3464567 , 0.6075949 , 0.17322835, 0.4177215 , 0.77165353, 0.2658228 , 0.85714287, 0.04724409, 0.71428573, 0.79527557, 0.2857143 , 0.5984252 , 1. , 0.35433072, 0.85714287, 0.15748031, 0.2857143 , 0.8110236 , 0.5714286 , 0.68503934, 0.85714287, 0.37795275, 0.5714286 , 0.17322835, 0.42857143, 0.88188976, 0.14285715, 0.6535433 , 0. , 0.42519686, 0.71428573, 0.24409449, 0.5714286 , 0.88188976, 0.5714286 , 0.7637795 , 0. , 0.503937 , 0.85714287, 0.27559054, 0.85714287, 0.08661418, 0.5714286 , 0.7401575 , 0.42857143, 0.5748032 , 0.85714287, 0.2992126 , 0.85714287, 0.08661418, 0.14285715, 0.7559055 , 0.14285715, 0.5511811 , 1. , 0.39370078, 0.85714287, 0.09448819, 0.5714286 , 0.8425197 , 0.42857143, 0.5590551 , 0.14285715, 0.39370078, 0.5714286 , 0.24409449, 0.5714286 , 0.92913383, 0.42857143, 0.6535433 , 0.2857143 , 0.51968503, 0.71428573, 0.27559054, 0.2857143 , 0.96850395, 0.2857143 , 0.7559055 , 0. , 0.48818898, 0.71428573, 0.26771653, 0.5714286 , 0.07874016, 0.5714286 , 0.7322835 , 0. , 0.54330707, 1. , 0.2519685 ][:,:]
