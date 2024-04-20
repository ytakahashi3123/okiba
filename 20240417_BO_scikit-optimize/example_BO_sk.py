import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from scipy.optimize import rosen

# 目的関数 (Rosenbrock関数)
def objective(x):
    return rosen(x)

# 最適化するパラメータの範囲
space = [Real(-2.0, 2.0, name='x1'),
         Real(-2.0, 2.0, name='x2')]

# ベイズ最適化の実行
res = gp_minimize(objective, space, n_calls=20, random_state=42)

# 結果の出力
print("最適な点:", res.x)
print("最小値:", res.fun)

# 収束のプロット
plot_convergence(res)
