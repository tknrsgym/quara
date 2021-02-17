#!/usr/bin/env python
# coding: utf-8

# In[1]:


# State


# In[5]:


import pickle
import time
from typing import List
import datetime as dt
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from tqdm import tqdm

from quara.data_analysis import data_analysis, physicality_violation_check, report
from quara.data_analysis.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)
from quara.data_analysis.weighted_relative_entropy import (
    WeightedRelativeEntropy,
    WeightedRelativeEntropyOption,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_y_measurement,
    get_z_povm,
)
from quara.objects.qoperation import QOperation
from quara.objects.state import State, get_z0_1q, get_z1_1q, get_x0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)
from quara.objects.operators import tensor_product

from quara.data_analysis.simulation import SimulationSetting


# In[6]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[7]:


start_all = time.time()


# In[8]:


# setup system
e_sys_1 = ElementalSystem(0, get_normalized_pauli_basis())
c_sys_1 = CompositeSystem([e_sys_1])

e_sys_2 = ElementalSystem(1, get_normalized_pauli_basis())
c_sys_2 = CompositeSystem([e_sys_2])

tester_objects = []

func_list = [get_x_povm, get_y_measurement, get_z_povm]

for i, funcs in enumerate(itertools.product(func_list, func_list)):
    povm1 = funcs[0](c_sys_1)
    povm2 = funcs[1](c_sys_2)

    povm_2qubit = tensor_product(povm1, povm2)
    tester_objects.append(povm_2qubit)


# In[9]:


# |0><0|
true_object = tensor_product(get_z0_1q(c_sys_1), get_z0_1q(c_sys_2))


# In[11]:


num_data = [100, 1000]
n_rep = 10

case_name_list = [
    "Linear(True)",
    "Linear(False)",
    "ProjectedLinear(True)",
    "ProjectedLinear(False)",
]

seed = 777
qtomography_list = [
    StandardQst(tester_objects, on_para_eq_constraint=True, seed=seed),
    StandardQst(tester_objects, on_para_eq_constraint=False, seed=seed),
    StandardQst(tester_objects, on_para_eq_constraint=True, seed=seed),
    StandardQst(tester_objects, on_para_eq_constraint=False, seed=seed),
]
para_list = [
    True,
    False,
    True,
    False,
]

estimator_list = [
    LinearEstimator(),
    LinearEstimator(),
    ProjectedLinearEstimator(),
    ProjectedLinearEstimator(),
]

estimation_results_list = []
elapsed_times = []
simulation_settings = []

for i, name in enumerate(case_name_list):
    qtomography = qtomography_list[i]
    estimator = estimator_list[i]

    start = time.time()

    estimation_results = data_analysis.estimate(
        qtomography=qtomography,
        true_object=true_object,
        num_data=num_data,
        estimator=estimator,
        iteration=n_rep,
    )

    # stock settings of this simulation
    simulation_setting = SimulationSetting(name=name, estimator=estimator)
    print(simulation_setting)
    simulation_settings.append(simulation_setting)

    estimation_results_list.append(estimation_results)

    elapsed_time = time.time() - start
    print("elapsed_time: {0}".format(elapsed_time / 60) + "[min]\n")
    elapsed_times.append(elapsed_time)


# In[ ]:

data = {
    "simulation_settings": simulation_settings,
    "estimation_results_list": estimation_results_list,
    "true_object": true_object,
    "tester_objects": tester_objects,
    "elapsed_times": elapsed_times,
    "seed": seed,
}

path = f"output_qst_2qubit_nrep={n_rep}/qst_2qubit_estimation_results_nrep={n_rep}_all.pickle"
Path(path).parent.mkdir(exist_ok=True)
with open(path, "wb") as f:
    pickle.dump(data, f)


# In[15]:


report.export_report(
    "qst_2qubit_nrep={n_rep}.pdf",
    estimation_results_list=estimation_results_list,  # 「EstimationResultのリスト」のリスト
    simulation_settings=simulation_settings,
    true_object=true_object,  # True Object
    tester_objects=tester_objects,  # Tester Objectのリスト.
    seed=seed,  # 推定で使ったseed（オプション）
    computation_time=sum(elapsed_times),  # 処理時間の合計（オプション）
)


# In[ ]:

all_elapsed_time = time.time() - start_all
print("elapsed_time(All): {0}".format(all_elapsed_time / 60) + "[min]\n")
