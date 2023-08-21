using Random: rand
using MLJModelInterface
using TimeSeriesClassification


@noinline pre(m, t) = begin
    MLJModelInterface.fit(m, 0, t)
end

# Precompile
pre(MiniRocketModel(shuffled=false), randn(10, 10));
pre(MiniRocketModel(shuffled=true), randn(10, 10));

ns = [10, 100, 500, 1000, 5000]
ms = [10, 100, 500, 1000, 5000]

print("Shuffled")
for n in ns, m in ms
    print("f$n $m")
    mini = MiniRocketModel(shuffled=true)
    @time for _ in 1:10
        pre(mini, randn(n, m))
    end
end

print("Not shuffled")
for n in ns, m in ms
    print("f$n $m")
    mini = MiniRocketModel(shuffled=false)
    @time for _ in 1:10
        pre(mini, randn(n, m))
    end
end
