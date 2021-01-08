#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Qpt


# In[9]:


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

import numpy.testing as npt

from quara.data_analysis import data_analysis, physicality_violation_check, report
from quara.data_analysis.projected_gradient_descent_base import (
    ProjectedGradientDescentBase,
    ProjectedGradientDescentBaseOption,
)
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
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)

from quara.objects.operators import tensor_product

from quara.data_analysis.simulation import SimulationSetting


# In[10]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[11]:


start_all = time.time()


# In[12]:


# setup system
e_sys_1 = ElementalSystem(0, get_normalized_pauli_basis())
c_sys_1 = CompositeSystem([e_sys_1])

e_sys_2 = ElementalSystem(1, get_normalized_pauli_basis())
c_sys_2 = CompositeSystem([e_sys_2])

# Tester States
tester_states = []

# |+><+| |+i><+i| |0><0| |1><1|
func_list = [get_x0_1q, get_y0_1q, get_z0_1q, get_z1_1q]

for i, funcs in enumerate(itertools.product(func_list, func_list)):
    state1 = funcs[0](c_sys_1)
    state2 = funcs[1](c_sys_2)

    state_2qubit = tensor_product(state1, state2)

    tester_states.append(state_2qubit)

# Tester POVMs
tester_povms = []

func_list = [get_x_measurement, get_y_measurement, get_z_measurement]

for i, funcs in enumerate(itertools.product(func_list, func_list)):
    povm1 = funcs[0](c_sys_1)
    povm2 = funcs[1](c_sys_2)

    povm_2qubit = tensor_product(povm1, povm2)
    tester_povms.append(povm_2qubit)


# In[5]:


# True Object
true_objects = []

i_gate_0 = get_depolarizing_channel(p=0, c_sys=c_sys_1)
i_gate_1 = get_depolarizing_channel(p=0, c_sys=c_sys_2)
gate_ii = tensor_product(i_gate_0, i_gate_1)

true_objects.append(gate_ii)

# true_objects.append(get_depolarizing_channel(p=0, c_sys=c_sys))
# true_objects.append(get_depolarizing_channel(p=0.05, c_sys=c_sys))
# true_objects.append(get_depolarizing_channel(p=1, c_sys=c_sys))
# true_objects.append(get_x_rotation(theta=np.pi/2, c_sys=c_sys))
# true_objects.append(get_x_rotation(theta=np.pi, c_sys=c_sys))
# true_objects.append(get_amplitutde_damping_channel(gamma=0.1, c_sys=c_sys))

true_object = true_objects[0]


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
    print("elapsed_time:{0}".format(elapsed_time / 60) + "[min]\n")
    elapsed_times.append(elapsed_time)


# In[ ]:

data = {
    "simulation_settings": simulation_settings,
    "estimation_results_list": estimation_results_list,
    "true_object": true_object,
    "tester_states": tester_states,
    "tester_povms": tester_povms,
    "elapsed_times": elapsed_times,
    "seed": seed,
}

path = f"output_qpt_2qubit_nrep={n_rep}/qpt_2qubit_estimation_results_nrep={n_rep}_all.pickle"
Path(path).parent.mkdir(exist_ok=True)
with open(path, "wb") as f:
    pickle.dump(data, f)


# In[7]:


report.export_report(
    f"qpt_2qubit_nrep={n_rep}.pdf",
    estimation_results_list,
    simulation_settings,
    true_object,
    tester_states + tester_povms,
    seed=seed,
    computation_time=sum(elapsed_times),
)


# In[14]:


all_elapsed_time = time.time() - start_all
print("elapsed_time(All): {0}".format(all_elapsed_time / 60) + "[min]\n")


# In[ ]:

