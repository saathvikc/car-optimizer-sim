from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
import numpy as np
import pandas as pd
import os

# Ensure output folder exists
OUTPUT_DIR = "./optimization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# Optimization Problem Wrapper
# -------------------
class DragMinimizationProblem(ElementwiseProblem):
    def __init__(self, model, X_template):
        self.model = model
        self.feature_names = X_template.columns.tolist()
        self.lb = X_template.min().values
        self.ub = X_template.max().values

        super().__init__(n_var=len(self.feature_names),
                         n_obj=1,
                         n_constr=0,
                         xl=self.lb,
                         xu=self.ub)

    def _evaluate(self, x, out, *args, **kwargs):
        pred = self.model.predict(np.array(x).reshape(1, -1))[0]
        out["F"] = pred

# -------------------
# Run Optimization
# -------------------
def run_drag_optimization(model, X_template, generations=100):
    problem = DragMinimizationProblem(model, X_template)
    algo = GA(pop_size=40)
    termination = get_termination("n_gen", generations)

    res = minimize(problem,
                   algo,
                   termination,
                   seed=42,
                   verbose=True)

    optimized_design = pd.Series(res.X, index=X_template.columns)
    optimized_cd = res.F[0] if res.F is not None else None

    optimized_design.to_csv(f"{OUTPUT_DIR}/optimized_design.csv")

    with open(f"{OUTPUT_DIR}/optimized_cd.txt", "w") as f:
        if optimized_cd is not None:
            f.write(f"Predicted Cd: {optimized_cd:.4f}\n")
        else:
            f.write("Optimization failed: No result for Cd.\n")

    return optimized_design, optimized_cd
