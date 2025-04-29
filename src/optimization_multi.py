# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

# %% [markdown]
# (batch_elution_optimization_multi)=
# # Optimize Batch Elution Process (Multi-Objective)
#
# ## Setup Optimization Problem

# %%
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('batch_elution_multi')

from examples.batch_elution.process import process
optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable('cycle_time', lb=10, ub=600)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)

optimization_problem.add_linear_constraint(
    ['feed_duration.time', 'cycle_time'], [1, -1]
)

# %% [markdown]
# ## Setup Simulator

# %%
from CADETProcess.simulator import Cadet
process_simulator = Cadet()
process_simulator.evaluate_stationarity = True

optimization_problem.add_evaluator(process_simulator)

# %% [markdown]
# ## Setup Fractionator

# %%
from CADETProcess.fractionation import FractionationOptimizer
frac_opt = FractionationOptimizer()

optimization_problem.add_evaluator(
    frac_opt,
    kwargs={
        'purity_required': [0.95, 0.95],
        'ignore_failed': False,
        'allow_empty_fractions': False,
    }
)

# %% [markdown]
# ## Setup Objectives

# %%
from CADETProcess.performance import Productivity, Recovery, EluentConsumption

productivity = Productivity()
optimization_problem.add_objective(
    productivity,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)

recovery = Recovery()
optimization_problem.add_objective(
    recovery,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)

eluent_consumption = EluentConsumption()
optimization_problem.add_objective(
    eluent_consumption,
    n_objectives=2,
    requires=[process_simulator, frac_opt],
    minimize=False,
)


# %% [markdown]
# ## Add callback for post-processing

# %%
def callback(fractionation, individual, evaluation_object, callbacks_dir):
    fractionation.plot_fraction_signal(
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_fractionation.png',
        show=False
    )

optimization_problem.add_callback(
    callback, requires=[process_simulator, frac_opt]
)

# %% [markdown]
# ## Configure Optimizer

# %%
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
optimizer.n_max_gen = 3
optimizer.pop_size = 3
optimizer.n_cores = 3

# %% [markdown]
# ## Run Optimization

# %%
if __name__ == '__main__':
    optimization_results = optimizer.optimize(
        optimization_problem,
        use_checkpoint=True
    )

# %% [markdown]
# ### Optimization Progress and Results
#
# The `OptimizationResults` which are returned contain information about the progress of the optimization.
# For example, the attributes `x` and `f` contain the final value(s) of parameters and the objective function.

# %% editable=true slideshow={"slide_type": ""} tags=["solution"]
print(optimization_results.x)
print(optimization_results.f)

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# After optimization, several figures can be plotted to vizualize the results.
# For example, the convergence plot shows how the function value changes with the number of evaluations.

# %% editable=true slideshow={"slide_type": ""} tags=["solution"]
optimization_results.plot_convergence()

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# The `plot_objectives` method shows the objective function values of all evaluated individuals.
# Here, lighter color represent later evaluations.
# Note that by default the values are plotted on a log scale if they span many orders of magnitude.
# To disable this, set `autoscale=False`.

# %% editable=true slideshow={"slide_type": ""} tags=["solution"]
optimization_results.plot_objectives()

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# Note that more figures are created for constrained optimization, as well as single-objective optimization.
# All figures are also saved automatically in the `working_directory`.
# Moreover, results are stored in a `.csv` file.
# - The `results_all.csv` file contains information about all evaluated individuals.
# - The `results_last.csv` file contains information about the last generation of evaluated individuals.
# - The `results_pareto.csv` file contains only the best individual(s).

# %%
