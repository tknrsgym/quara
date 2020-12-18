#!/usr/bin/env python
# coding: utf-8

# In[1]:


# POVMT


# In[2]:


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
from quara.objects.state import (
    State,
    get_x0_1q,
    get_x1_1q,
    get_y0_1q,
    get_y1_1q,
    get_z0_1q,
    get_z1_1q,
)
from quara.objects.povm import (
    Povm,get_x_measurement
)
from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)


# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


start_all = time.time()


# In[5]:


# setup system
e_sys = ElementalSystem(0, get_normalized_pauli_basis())
c_sys = CompositeSystem([e_sys])

# |+><+|
state_x0 = get_x0_1q(c_sys)
# |+i><+i|
state_y0 = get_y0_1q(c_sys)
# |0><0|
state_z0 = get_z0_1q(c_sys)
# |1><1|
state_z1 = get_z1_1q(c_sys)
tester_objects = [state_x0, state_y0, state_z0, state_z1]


# In[9]:


true_objects = []

# Case 1: 軸上
a0, a1, a2, a3 = 1, 1, 0, 0
m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])
true_object = Povm(vecs=[m1, m2], c_sys=c_sys)

true_objects.append(true_object)

# Case 2: 軸から一番離れた表面
a0, a1, a2, a3 = 1, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)
m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])
true_object = Povm(vecs=[m1, m2], c_sys=c_sys)

true_objects.append(true_object)

# Case 3: 原点
a0, a1, a2, a3 = 1, 0, 0, 0
m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])
true_object = Povm(vecs=[m1, m2], c_sys=c_sys)

true_objects.append(true_object)

true_object_names = ["軸上", "軸から一番離れた表面", "原点"]


# In[10]:


num_data = [100, 1000]
n_rep = 10
measurement_n = len(true_object.vecs)  # 測定値の数

case_name_list = [
    "LinearEstimator(True)",
    "LinearEstimator(False)",
    "ProjectedLinearEstimator(True)",
    "ProjectedLinearEstimator(False)",
]

seed = 777
qtomography_list = [
    StandardPovmt(tester_objects, measurement_n, on_para_eq_constraint=True, seed=seed),
    StandardPovmt(
        tester_objects, measurement_n, on_para_eq_constraint=False, seed=seed
    ),
    StandardPovmt(tester_objects, measurement_n, on_para_eq_constraint=True, seed=seed),
    StandardPovmt(
        tester_objects, measurement_n, on_para_eq_constraint=False, seed=seed
    ),
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
output_dir = f"output_povmt_nrep={n_rep}"
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

    # Result
    result = dict(
        estimation_results_list=estimation_results_list,
        true_object=true_object,
        elapsed_times=elapsed_times,
    )
    all_results.append(result)
    # output
    with open(
        f"{output_dir}/pickle/povmt_estimation_results_nrep={n_rep}_{true_object_names[true_idx]}.pickle",
        "wb",
    ) as f:
        pickle.dump(result, f)
    # PDF report
    path = (
        f"{output_dir}/pdf/sample_povmt_case{true_idx}_{true_object_names[true_idx]}.pdf"
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
        f"{output_dir}/mse/sample_povmt_case{true_idx}_{true_object_names[true_idx]}.html"
    )
    fig.write_html(path)


with open(f"{output_dir}/povmt_estimation_results_nrep={n_rep}_all.pickle", "wb") as f:
    pickle.dump(all_results, f)

print("Completed.")


# In[ ]:




