#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Povm


# In[1]:


import pickle
import time
from typing import List
import datetime as dt
from pathlib import Path

import numpy as np

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
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.gate import (
    Gate,
    get_depolarizing_channel,
    get_x_rotation,
    get_amplitutde_damping_channel,
)
from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)
import pickle


# In[2]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[4]:


# setup system
e_sys = ElementalSystem(0, get_normalized_pauli_basis())
c_sys = CompositeSystem([e_sys])

# Tester Objects (State)
# |+><+|
state_x0 = get_x0_1q(c_sys)
# |+i><+i|
state_y0 = get_y0_1q(c_sys)
# |0><0|
state_z0 = get_z0_1q(c_sys)
# |1><1|
state_z1 = get_z1_1q(c_sys)
tester_states = [state_x0, state_y0, state_z0, state_z1]


# In[5]:


tester_povms = [
    get_x_measurement(c_sys),
    get_y_measurement(c_sys),
    get_z_measurement(c_sys),
]


# In[8]:


# True Object
true_objects = []
true_objects.append(get_depolarizing_channel(p=0, c_sys=c_sys))
true_objects.append(get_depolarizing_channel(p=0.05, c_sys=c_sys))
true_objects.append(get_depolarizing_channel(p=1, c_sys=c_sys))
true_objects.append(get_x_rotation(theta=np.pi / 2, c_sys=c_sys))
true_objects.append(get_x_rotation(theta=np.pi, c_sys=c_sys))
true_objects.append(get_amplitutde_damping_channel(gamma=0.1, c_sys=c_sys))


# In[16]:


true_object_names = [
    "depolarizing_channel_p=0",
    "depolarizing_channel_p=0.05",
    "depolarizing_channel_p=1",
    "x_rotation_theta=np.pi_half",
    "x_rotation_theta=np.pi",
    "amplitutde_damping_channel_gamma=0.1",
]


# In[9]:


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
    StandardQpt(tester_states, tester_povms, on_para_eq_constraint=True, seed=seed),
    StandardQpt(tester_states, tester_povms, on_para_eq_constraint=False, seed=seed),
    StandardQpt(tester_states, tester_povms, on_para_eq_constraint=True, seed=seed),
    StandardQpt(tester_states, tester_povms, on_para_eq_constraint=False, seed=seed),
]
para_list = [True, False, True, False]

estimator_list = [
    LinearEstimator(),
    LinearEstimator(),
    ProjectedLinearEstimator(),
    ProjectedLinearEstimator(),
]

all_results = []
output_dir = f"output_qpt_nrep={n_rep}"
Path(output_dir).mkdir(exist_ok=True)
Path(f"{output_dir}/pickle").mkdir(exist_ok=True)
Path(f"{output_dir}/pdf").mkdir(exist_ok=True)
Path(f"{output_dir}/mse").mkdir(exist_ok=True)

for true_idx, true_object in enumerate(true_objects):
    print("====================================")
    print(f"{true_idx}: {true_object_names[true_idx]}, {true_object}")
    print("====================================")
    estimation_results_list = []
    elapsed_times = []

    for case_idx, name in enumerate(case_name_list):
        qtomography = qtomography_list[case_idx]
        estimator = estimator_list[case_idx]

        start = time.time()
        print(f"Case {case_idx}: {name}")
        print(f"Parametorization: {para_list[case_idx]}")
        print(f"Type of qtomography: {qtomography.__class__.__name__}")
        print(f"Estimator: {estimator.__class__.__name__}")

        print(f"true_object={type(true_object)}")
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
        f"{output_dir}/pickle/qpt_estimation_results_nrep={n_rep}_{true_object_names[true_idx]}.pickle",
        "wb",
    ) as f:
        pickle.dump(result, f)
    # PDF report
    path = (
        f"{output_dir}/pdf/sample_qpt_case{true_idx}_{true_object_names[true_idx]}.pdf"
    )
    report.export_report(
        path,
        result["estimation_results_list"],
        case_name_list,
        estimator_list,
        result["true_object"],
        tester_states + tester_povms,
        seed=seed,
        computation_time=sum(result["elapsed_times"]),
        show_physicality_violation_check=False,
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
        f"{output_dir}/mse/sample_qpt_case{true_idx}_{true_object_names[true_idx]}.html"
    )
    fig.write_html(path)


with open(f"{output_dir}/qpt_estimation_results_nrep={n_rep}_all.pickle", "wb") as f:
    pickle.dump(all_results, f)

print("Completed.")
# In[ ]:

