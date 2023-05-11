from sktime.transformations.panel.rocket._minirocket import MiniRocket
from sktime.transformations.panel.rocket._minirocket_numba import _fit, _transform
import numpy as np
import tracemalloc
import numba.core.runtime

DONT_PRINT_STUFF = True

ns = [1, 2, 3, 4, 7, 11, 18, 29, 46, 75, 121, 196, 316, 511, 825, 1334, 2154, 3481, 5623, 9085, 14678, 23714, 38312, 61897, 100000]
ms = [9, 11, 18, 29, 46, 75, 121, 196, 316, 511, 825, 1334, 2154, 3481]

print('N,M,fit,transform')
for N in ns:
    for M in ms:
        DATA = np.random.randn(N,M).astype(np.float32)

        _mr = MiniRocket(num_kernels=84*4)
        _p = _fit(DATA, _mr.num_kernels, _mr.max_dilations_per_kernel)
        _t = _transform(DATA, _p)

        tracemalloc.stop()
        tracemalloc.clear_traces()
        tracemalloc.start()

        s1 = tracemalloc.take_snapshot()
        t1 = s1.statistics('lineno')
        n1 = numba.core.runtime.rtsys.get_allocation_stats()

        mr = MiniRocket(num_kernels=84*4)

        s2 = tracemalloc.take_snapshot()
        t2 = s2.statistics('lineno')
        n2 = numba.core.runtime.rtsys.get_allocation_stats()

        p = _fit(DATA, mr.num_kernels, mr.max_dilations_per_kernel)

        s3 = tracemalloc.take_snapshot()
        t3 = s3.statistics('lineno')
        n3 = numba.core.runtime.rtsys.get_allocation_stats()

        _transform(DATA, p)

        s4 = tracemalloc.take_snapshot()
        t4 = s4.statistics('lineno')
        n4 = numba.core.runtime.rtsys.get_allocation_stats()

        tracemalloc.stop()

        if not DONT_PRINT_STUFF: print("=============== 1  |", n1)
        for stat in t1:
            if DONT_PRINT_STUFF: break
            if 'sktime' not in str(stat):
                continue
            print(stat)

        if not DONT_PRINT_STUFF: print("=============== 2  |", [x-y for x, y in zip (n2, n1)])
        for stat in t2:
            if DONT_PRINT_STUFF: break
            if 'sktime' not in str(stat):
                continue
            print(stat)

        if not DONT_PRINT_STUFF: print("=============== 3  |", [x-y for x, y in zip (n3, n2)])
        for stat in t3:
            if DONT_PRINT_STUFF: break
            if 'sktime' not in str(stat):
                continue
            print(stat)

        if not DONT_PRINT_STUFF: print("=============== 4  |", [x-y for x, y in zip (n4, n3)])
        for stat in t4:
            if DONT_PRINT_STUFF: break
            if 'sktime' not in str(stat):
                continue
            print(stat)
        
        print(f'{N},{M},{n3[0]-n2[0]},{n4[0]-n3[0]}')
