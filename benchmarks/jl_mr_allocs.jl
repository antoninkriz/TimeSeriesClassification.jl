using Random: rand
using MLJModelInterface
using TimeSeriesClassification

pre(N, M, p) = begin
    p && print("$N,$M")
    trainX = randn(N, M)
    mini = MiniRocketModel()
    fp = MLJModelInterface.fit(mini, 0, trainX)
    if p
        @time MLJModelInterface.transform(mini, fp[1], trainX)
    else
        MLJModelInterface.transform(mini, fp[1], trainX)
    end
    return
end
pre(10,10,false)


ns = [9, 11, 18, 29, 46, 75, 121, 196, 316, 511, 825, 1334, 2154, 3481]
ms = [1, 2, 3, 4, 7, 11, 18, 29, 46, 75, 121, 196, 316, 511, 825, 1334, 2154, 3481, 5623, 9085, 14678, 23714, 38312, 61897, 100000]
for n in ns, m in ms
    pre(n,m,true)
end
