#!/usr/bin/env python
# coding: utf-8

# In[5]:


# QST


# In[1]:


import pickle
import time
from typing import List
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from tqdm import tqdm

from quara.data_analysis import data_analysis, physicality_violation_check, report
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.qoperation import QOperation
from quara.objects.state import State, get_z0_1q, get_z1_1q, get_x0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)

import pickle


# In[2]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[3]:


start_all = time.time()


# In[4]:


# setup system
e_sys = ElementalSystem(0, get_normalized_pauli_basis())
c_sys = CompositeSystem([e_sys])

povm_x = get_x_measurement(c_sys)
povm_y = get_y_measurement(c_sys)
povm_z = get_z_measurement(c_sys)
tester_objects = [povm_x, povm_y, povm_z]


# In[5]:


true_objects = []

# Case 1:
true_object = get_z0_1q(c_sys)
true_objects.append(true_object)

# Case 2:
vec = np.array(
    [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(6), 1 / np.sqrt(6)], dtype=np.float64
)
true_object = State(c_sys, vec)
true_objects.append(true_object)

# Case 3:
vec = np.array([1 / np.sqrt(2), 0, 0, 0], dtype=np.float64)
true_object = State(c_sys, vec)
true_objects.append(true_object)

true_object_names = ["軸上", "軸から一番離れた表面", "原点"]


# In[6]:


num_data = [100, 1000]
n_rep = 10

case_name_list = [
    "LinearEstimator(True)",
    "LinearEstimator(False)",
    "ProjectedLinearEstimator(True)",
    "ProjectedLinearEstimator(False)",
]

seed = 777
qtomography_list = [
    StandardQst(tester_objects, on_para_eq_constraint=True, seed=seed),
    StandardQst(tester_objects, on_para_eq_constraint=False, seed=seed),
    StandardQst(tester_objects, on_para_eq_constraint=True, seed=seed),
    StandardQst(tester_objects, on_para_eq_constraint=False, seed=seed),
]
para_list = [True, False, True, False]

estimator_list = [
    LinearEstimator(),
    LinearEstimator(),
    ProjectedLinearEstimator(),
    ProjectedLinearEstimator(),
]

# estimation_results_list = []
# elapsed_times = []

all_results = []
output_dir = f"output_qst_nrep={n_rep}"
Path(output_dir).mkdir(exist_ok=True)
Path(f"{output_dir}/pickle").mkdir(exist_ok=True)
Path(f"{output_dir}/pdf").mkdir(exist_ok=True)
Path(f"{output_dir}/mse").mkdir(exist_ok=True)

for true_idx, true_object in enumerate(true_objects):
    print("====================================")
    print(f"{true_idx}: {true_object_names[true_idx]}")
    print("True Object:")
    print(f"{true_object}")
    print("====================================")
    estimation_results_list = []
    elapsed_times = []

    for i, name in enumerate(case_name_list):
        qtomography = qtomography_list[i]
        estimator = estimator_list[i]

        start = time.time()
        print(f"Case {i}: {name}")
        print(f"Parametorization: {para_list[i]}")
        print(f"Type of qtomography: {qtomography.__class__.__name__}")
        print(f"Estimator: {estimator.__class__.__name__}")

        estimation_results = data_analysis.estimate(
            qtomography=qtomography,
            true_object=true_object,
            num_data=num_data,
            estimator=estimator,
            iteration=n_rep,
        )
        estimation_results_list.append(estimation_results)

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time / 60) + "[min]\n")
        elapsed_times.append(elapsed_time)

    result = dict(
        estimation_results_list=estimation_results_list,
        true_object=true_object,
        elapsed_times=elapsed_times,
    )
    all_results.append(result)
    # output
    with open(
        f"{output_dir}/pickle/qst_estimation_results_nrep={n_rep}_{true_object_names[true_idx]}.pickle",
        "wb",
    ) as f:
        pickle.dump(result, f)
    # PDF report
    path = (
        f"{output_dir}/pdf/sample_qst_case{true_idx}_{true_object_names[true_idx]}.pdf"
    )
    report.export_report(
        path,
        result["estimation_results_list"],
        case_name_list,
        estimator_list,
        result["true_object"],
        tester_objects,
        seed=seed,
        computation_time=sum(result["elapsed_times"]),
    )
    # Fig
    fig = data_analysis.make_mses_graph_estimation_results(
        result["estimation_results_list"],
        case_name_list,
        true_objects[true_idx],
        show_analytical_results=True,
        estimator_list=estimator_list,
    )
    path = (
        f"{output_dir}/mse/sample_qst_case{true_idx}_{true_object_names[true_idx]}.html"
    )
    fig.write_html(path)


with open(f"{output_dir}/qst_estimation_results_nrep={n_rep}_all.pickle", "wb") as f:
    pickle.dump(all_results, f)

print("Completed.")


# In[7]:


elapsed_time = time.time() - start_all
print("elapsed_time:{0}".format(elapsed_time / 60) + "[min]\n")


# In[ ]:

