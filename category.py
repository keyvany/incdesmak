#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 04:21:24 2022

@author: keyvanyahya
"""

from ANNarchy import (
    Neuron,
    Population,
    Projection,
    Synapse,
    setup,
    compile,
    Uniform,
    simulate,
    
)
import numpy as np
import random
from CompNeuroPy import Monitors, get_full_model, plot_recordings


### Setup ANNarchy
setup(dt=1.0, num_threads=1)

### General parameters
ITdimension = 10
baseline_it = random.uniform(0.0, 0.1)
response_threshold = 0.5
nr_fish_presentations = 100

### you have to present multiple times a fish
### --> fish_list contains the fish variants you can present to the model
fish_list = [
    [
        [-1.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, 1.0],
        [-0.5, 0.5, 0.0, 0.0],
        [-1, 0.5, 0.0, 1],
        [-0.5, -0.5, 1.0, 0.0],
        [-0.5, -1, 0.0, 1.0],
    ],
    [
        [1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.5, -0.5, 0.0, -1.0],
        [1, -0.5, 0.0, -1.0],
        [0.5, 0.5, 1.0, 0.0],
        [0.5, 1, 0.0, 1.0],
    ],
]
feature_name_list = ["DF", "TF", "VF", "MA"]

### define the durations of the events
trial_duration = 200
dopamine_input_duration = 300
deactivate_input_duration = 30
calcium_trace_decline_duration = 100

### Set the dopamine condition
max_dopa = 1.0
baseline_dopa = 0.1
baseline_snc = 0.1
min_dopa = 0.0
K_dip = 0.4


### Neuron models
# In Fran's version SaturatedNeuron=LinearNeuron and NormalizationNeuron=LinearNeuron and LinearNeuronPFC=LinearNeuron
LinearNeuron = Neuron(
    parameters="""
        tau = 10.0
        baseline = 0.0
        noise = 0.0
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = pos(mp)
    """,
)

DopamineNeuron = Neuron(
    parameters="""
        tau = 10.0
        firing = 0
        inhibition = 0.0
        baseline = 0.0
    """,
    equations="""
        ex_in = if (sum(exc)>0): 1 else: 0
        s_inh = sum(inh)
        aux = if (firing>0): (ex_in)*(pos(1.0-baseline-s_inh) + baseline) + (1-ex_in)*(-10*sum(inh)+baseline)  else: baseline
        tau*dmp/dt + mp =  aux
        r = if (mp>0.0): mp else: 0.0
    """,
)

InputNeuron = Neuron(
    parameters="""
        tau = 1.5
        baseline = 0.0
    """,
    equations="""
        tau*dmp/dt + mp = baseline
        baseline_rec = baseline
        r = if (mp>0.0): mp else: 0.0
    """,
)

### Synapse models
# ?? population?
PostCovariance = Synapse(
    parameters="""
        tau = 1000.0
        tau_alpha = 10.0
        regularization_threshold = 1.0
        threshold_post = 0.0
        threshold_pre = 0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha =  pos(post.mp - regularization_threshold)


        trace = (pre.r - mean(pre.r) - threshold_pre) * pos(post.r - mean(post.r) - threshold_post)
    delta = (trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post)*w)
        tau*dw/dt = delta : min=0
   """,
)

ReversedSynapse = Synapse(
    parameters="""
        reversal = 1.0
    """,
    psp="""
        w*pos(reversal-pre.r)
    """,
)

# DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
DAPostCovarianceNoThreshold = Synapse(
    parameters="""
        tau=1000.0
        tau_alpha=10.0
        regularization_threshold=1.0
        baseline_dopa = 0.1
        K_burst = 1.0
        K_dip = 0.4
        DA_type = 1
        threshold_pre=0.0
        threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)

        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)

    condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0

        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: condition_0*DA_type*K_dip*dopa_sum

        

        delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
        tau*dw/dt = delta : min=0
    """,
)

# Excitatory synapses STN -> GPi
DAPreCovariance_excitatory = Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0
    regularization_threshold=1.0
    baseline_dopa = 0.1
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  = pos( post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)

        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - threshold_post)
        aux = if (trace<0.0): 1 else: 0
        dopa_mod = if (dopa_sum>0): K_burst * dopa_sum else: K_dip * dopa_sum * aux
        delta = dopa_mod * trace - alpha * pos(trace)
        tau*dw/dt = delta : min=0

        
    """,
)

# Inhibitory synapses GPi -> GPi and STRD2 -> GPe
DAPreCovariance_inhibitory = Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0
    regularization_threshold=1.0
    baseline_dopa = 0.1
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)

        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
        aux = if (trace>0): 1 else: 0

        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum

        delta = dopa_mod * trace - alpha * pos(trace)
        tau*dw/dt = delta : min=0
    """,
)

DAPrediction = Synapse(
    parameters="""
        tau = 100000.0
        baseline_dopa = 0.1
   """,
    equations="""
       aux = if (post.sum(exc)>0): 1.0 else: 3.0
       delta = aux*(post.r - baseline_dopa)*pos(pre.r - mean(pre.r))
       tau*dw/dt = delta : min=0
   """,
)


### Populations
# IT Input
IT = Population(name="IT", geometry=(ITdimension, ITdimension), neuron=InputNeuron)
IT.tau = 10.0
IT.baseline = baseline_it

# Reward Input
Reward_Layer = Population(name="Reward_Layer", geometry=1, neuron=InputNeuron)
Reward_Layer.tau = 1.0

# PFC_MTL
MTL = Population(name="PFC_MTL", geometry=(4, 9), neuron=LinearNeuron)
MTL.tau = 10.0
MTL.noise = 0.05
MTL.baseline = 0.0

# SNc
SNc = Population(name="SNc", geometry=1, neuron=DopamineNeuron)
SNc.tau = 10.0
SNc.firing = 0
SNc.baseline = baseline_snc

# Striatum direct pathway
StrD1 = Population(name="StrD1", geometry=(4, 4), neuron=LinearNeuron)
StrD1.tau = 10.0
StrD1.noise = 0.1
StrD1.baseline = 0.4

# Striatum indirect pathway
StrD2 = Population(name="StrD2", geometry=(4, 4), neuron=LinearNeuron)
StrD2.tau = 10.0
StrD2.noise = 0.1
StrD2.baseline = 0.4

# Striatum feedback pathway
StrThal = Population(name="StrThal-PFC", geometry=2, neuron=LinearNeuron)
StrThal.tau = 10.0
StrThal.noise = 0.1
StrThal.baseline = 0.4

# GPi
GPi = Population(name="GPi-PFC", geometry=2, neuron=LinearNeuron)
GPi.tau = 10.0
GPi.noise = 1.0
GPi.baseline = 2.4

# STN
STN = Population(name="STN", geometry=(4, 4), neuron=LinearNeuron)
STN.tau = 10.0
STN.noise = 0.1
STN.baseline = 0.4

# GPe
GPe = Population(name="GPe", geometry=2, neuron=LinearNeuron)
GPe.tau = 10.0
GPe.noise = 1.0
GPe.baseline = 1.0

# VA
VA = Population(name="VA-PFC", geometry=2, neuron=LinearNeuron)
VA.tau = 10.0
VA.noise = 0.0001
VA.baseline = 0.0

# PM
PM = Population(name="PM", geometry=2, neuron=LinearNeuron)
PM.tau = 10.0
PM.noise = 1.0


### Projections
ITMTL = Projection(pre=IT, post=MTL, target="exc", synapse=PostCovariance)
ITMTL.connect_all_to_all(weights=Uniform(0.2, 0.4))  # Normal(0.3,0.1) )
ITMTL.tau = 15000
ITMTL.regularization_threshold = 3.5
ITMTL.tau_alpha = 1.0
ITMTL.baseline_dopa = baseline_dopa
ITMTL.threshold_post = 0.0
ITMTL.thrshold_pre = 0.15

MTLVA_11 = Projection(pre=MTL[0:8], post=VA[0], target="exc")
MTLVA_11.connect_all_to_all(weights=0.15)
MTLVA_22 = Projection(pre=MTL[8:16], post=VA[1], target="exc")
MTLVA_22.connect_all_to_all(weights=0.15)


IT_StrD1 = Projection(
    pre=IT, post=StrD1, target="exc", synapse=DAPostCovarianceNoThreshold
)
IT_StrD1.connect_all_to_all(weights=Uniform(0, 0.3))  # Normal(0.15,0.15))
IT_StrD1.tau = 75.0
IT_StrD1.regularization_threshold = 1.0
IT_StrD1.tau_alpha = 1.0
IT_StrD1.baseline_dopa = baseline_dopa
IT_StrD1.K_dip = 0.4
IT_StrD1.K_burst = 1.0
IT_StrD1.DA_type = 1
IT_StrD1.threshold_pre = 0.15

IT_StrD2 = Projection(
    pre=IT, post=StrD2, target="exc", synapse=DAPostCovarianceNoThreshold
)
IT_StrD2.connect_all_to_all(weights=Uniform(0, 0.3))  # Normal(0.15,0.15))
IT_StrD2.tau = 75.0
IT_StrD2.regularization_threshold = 1.0
IT_StrD2.tau_alpha = 1.0
IT_StrD2.baseline_dopa = baseline_dopa
IT_StrD2.K_dip = 0.4
IT_StrD2.K_burst = 1.0
IT_StrD2.DA_type = -1
IT.threshold_pre = 0.15

IT_STN = Projection(pre=IT, post=STN, target="exc", synapse=DAPostCovarianceNoThreshold)
IT_STN.connect_all_to_all(weights=Uniform(0, 0.3))  # Normal(0.15,0.15))
IT_STN.tau = 75.0
IT_STN.regularization_threshold = 1.0
IT_STN.tau_alpha = 1.0
IT_STN.baseline_dopa = baseline_dopa
IT_STN.K_dip = 0.4
IT_STN.K_burst = 1.0
IT_STN.DA_type = 1
IT_STN.threshold_pre = 0.15

Reward_LayerSNc = Projection(pre=Reward_Layer, post=SNc, target="exc")
Reward_LayerSNc.connect_all_to_all(weights=1.0)

VAMTL_11 = Projection(pre=VA[0], post=MTL[0:8], target="exc")
VAMTL_11.connect_all_to_all(weights=0.35)
VAMTL_22 = Projection(pre=VA[1], post=MTL[8:16], target="exc")
VAMTL_22.connect_all_to_all(weights=0.35)

VAPM = Projection(pre=VA, post=PM, target="exc")
VAPM.connect_one_to_one(weights=1.0)

StrD1StrD1 = Projection(pre=StrD1, post=StrD1, target="inh")
StrD1StrD1.connect_all_to_all(weights=0.3)

STNSTN = Projection(pre=STN, post=STN, target="inh")
STNSTN.connect_all_to_all(weights=0.3)

MTLMTL = Projection(pre=MTL, post=MTL, target="inh")
MTLMTL.connect_all_to_all(weights=0.1)

PMPM = Projection(pre=PM, post=PM, target="inh")
PMPM.connect_all_to_all(weights=1.0)

StrD2StrD2 = Projection(pre=StrD2, post=StrD2, target="inh")
StrD2StrD2.connect_all_to_all(weights=0.3)

StrThalStrThal = Projection(pre=StrThal, post=StrThal, target="inh")
StrThalStrThal.connect_all_to_all(weights=0.3)

GPi_GPi = Projection(pre=GPi, post=GPi, target="exc", synapse=ReversedSynapse)
GPi_GPi.connect_all_to_all(weights=1.0)

StrD1_GPi = Projection(
    pre=StrD1, post=GPi, target="inh", synapse=DAPreCovariance_inhibitory
)
StrD1_GPi.connect_all_to_all(weights=Uniform(0, 0.05))  # Normal(0.025,0.025))
StrD1_GPi.tau = 50.0
StrD1_GPi.regularization_threshold = 1.0
StrD1_GPi.tau_alpha = 1.0
StrD1_GPi.baseline_dopa = baseline_dopa
StrD1_GPi.K_dip = 0.4
StrD1_GPi.threshold_post = 0.15
StrD1_GPi.DA_type = 1

STN_GPi = Projection(
    pre=STN, post=GPi, target="exc", synapse=DAPreCovariance_excitatory
)
STN_GPi.connect_all_to_all(weights=Uniform(0, 0.05))  # Normal(0.025,0.025))
STN_GPi.tau = 50.0
STN_GPi.regularization_threshold = 2.6
STN_GPi.tau_alpha = 1.0
STN_GPi.baseline_dopa = baseline_dopa
STN_GPi.K_dip = 0.4
STN_GPi.thresholdpost = -0.15
STN_GPi.DA_type = 1

StrD2_GPe = Projection(
    pre=StrD2, post=GPe, target="inh", synapse=DAPreCovariance_inhibitory
)
StrD2_GPe.connect_all_to_all(weights=Uniform(0, 0.05))  # Normal(0.025,0.025))
StrD2_GPe.tau = 50.0
StrD2_GPe.regularization_threshold = 2.0
StrD2_GPe.tau_alpha = 1.0
StrD2_GPe.baseline_dopa = baseline_dopa
StrD2_GPe.K_dip = 0.4
StrD2_GPe.threshold_post = 0.15
StrD2_GPe.DA_type = -1

StrD1_SNc = Projection(pre=StrD1, post=SNc, target="inh", synapse=DAPrediction)
StrD1_SNc.connect_all_to_all(weights=0.0)
StrD1_SNc.tau = 100000
StrD1_SNc.baseline_dopa = baseline_dopa

GPe_GPi = Projection(pre=GPe, post=GPi, target="inh")
GPe_GPi.connect_one_to_one(weights=1.5)

VAStr_Thal = Projection(pre=VA, post=StrThal, target="exc")
VAStr_Thal.connect_one_to_one(weights=1.0)

StrThal_GPe = Projection(pre=StrThal, post=GPe, target="inh")
StrThal_GPe.connect_one_to_one(weights=0.3)

StrThal_GPi = Projection(pre=StrThal, post=GPi, target="inh")
StrThal_GPi.connect_one_to_one(weights=0.3)

GPi_VA = Projection(pre=GPi, post=VA, target="inh")
GPi_VA.connect_one_to_one(weights=0.5)

SNc_StrD1 = Projection(pre=SNc, post=StrD1, target="dopa")
SNc_StrD1.connect_all_to_all(weights=1.0)

SNc_StrD2 = Projection(pre=SNc, post=StrD2, target="dopa")
SNc_StrD2.connect_all_to_all(weights=1.0)

SNc_GPi = Projection(pre=SNc, post=GPi, target="dopa")
SNc_GPi.connect_all_to_all(weights=1.0)

SNc_STN = Projection(pre=SNc, post=STN, target="dopa")
SNc_STN.connect_all_to_all(weights=1.0)

SNc_GPe = Projection(pre=SNc, post=GPe, target="dopa")
SNc_GPe.connect_all_to_all(weights=1.0)

SNc_MTL = Projection(pre=SNc, post=MTL, target="dopa")
SNc_MTL.connect_all_to_all(weights=1.0)

SNc_VA = Projection(pre=SNc, post=VA, target="dopa")
SNc_VA.connect_all_to_all(weights=1.0)


compile()


### Monitors
populations = get_full_model()["populations"]
mon_dict = {"pop;" + pop: ["r"] for pop in populations}
print(populations)
print(mon_dict)
mon_dict["pop;IT"] = ["r"]
#mon = Monitors(mon_dict)
#mon1=Monitors(["IT"])
mon=Monitors(mon_dict)
print(mon.monDict)
### Simulation functions
def get_fish(fish_list, feature_name_list):
    """
    return a dictionary with features and feature values

    Args:
        fish_list: list
            list with all possible fishes consisting of 4 feature values
        feature_name_list: list
            list containing 4 strings, the names of the 4 features (the order corresponds to the order in fish_list)
    """
    input_1 = random.randint(0, 1)
    # print(input_1)
    input_11 = random.randint(
        0, 5
    )  # -oliver-: why only 0,2 fish_list has 6 entries --> 0,5
    # print(input_11)
    # input_111 = random.randrange(0, 3, 1)  # --feature selection  -oliver-: input_111 not used
    # print(input_111)  # --feature selected
    input_sacc = fish_list[input_1][input_11]
    # print(input_sacc)
    fish = dict(zip(feature_name_list, input_sacc))
    return fish


def get_true_category(fish):
    """
    returns the category of a given fish based on a single feature
    """
    important_feature = "DF"
    if fish[important_feature] == 1 or fish[important_feature] == 0.5:
        cat = "B"
    if fish[important_feature] == -1 or fish[important_feature] == -0.5:
        cat = "A"
    return cat


def select_input(feature_name, fish, IT, feature_name_list, baseline_default):
    """
    activates IT neurons based on the presented feature

    Args:
        feature_name: string
            name of the activated feature
        fish: dictionary
            dictionary containing the feature values for the fish features
        IT: ANNarchy population
            the IT population
        feature_name_list: list
            list containing 4 strings, the names of the 4 features (the order corresponds to the order in fish_list)
        baseline_default: float
            default baseline of IT
    """
    ### set deafult baseline
    IT.baseline = baseline_default

    ### feature name i.e. spatial position defines the group of IT neurons
    nr_IT_neurons = len(IT)
    nr_group_neurons = nr_IT_neurons // (len(feature_name_list))
    neuron_group_idx = feature_name_list.index(feature_name)
    neuron_group = IT[
        int(neuron_group_idx * nr_group_neurons) : int(
            (neuron_group_idx + 1) * nr_group_neurons
        )
    ]

    ### feature value defines which neurons of neuron group get active --> neuron groups are feature-value selective
    feature_value = fish[feature_name]
    activity_center_idx = ((feature_value - (-1)) / (1 - (-1))) * (nr_group_neurons - 1)

    activity = gauss_1D(
        m=activity_center_idx,
        sig=nr_group_neurons / 10,
        size=nr_group_neurons,
        min=neuron_group.baseline[0],
    )

    ### set the activity of the neuron group
    neuron_group.baseline = activity


def gauss_1D(m, sig, size, min):
    """
    returns a 1D gaussian array

    Args:
        m: float
            center/mean of the gaussian along the array indizes
        sig: float
            standard deviation in array indizes
        size: int
            size of the returned array
        min: float
            minimum value
    """
    return np.clip(np.exp(-((np.arange(size) - m) ** 2) / (2 * sig**2)), min, None)


def get_feature_order(feature_name_list):
    """
    return a list with feature names containing the presentation order of the fish features
    """
    idx_arr = np.arange(len(feature_name_list))
    feature_name_arr = np.array(feature_name_list)
    np.random.shuffle(idx_arr)
    return feature_name_arr[idx_arr].tolist()


### Simulation
### start the loop over the fishes
consecutive_correct = 0
for _ in range(nr_fish_presentations):
    ### stop the presentation of fishes if 300 fishes were consecutively correctly classified
    if consecutive_correct == 300:
        break

    ### each fish is represented by 4 features which are presented sequentially (I think in random order)
    ### --> here the function get_features() somehow returns a list with the 4 features for a given fish
    ### and the function get_category returns the corresponding category of the fish
    fish = get_fish(fish_list, feature_name_list)
    feature_order = get_feature_order(feature_name_list)
    category_true = get_true_category(fish)

    ### loop over all features of the fish
    for feature_name in feature_order:
        mon.start()
        ### select the input = activate one feature of the fish in IT or MTL (the manuscript says IT)
        ### I think, based on the feature you have to activate specific neurons in the IT
        ### this is done by the select_input function
        select_input(feature_name, fish, IT, feature_name_list, baseline_it)

        ### simulate the duration of the trial / how long the feature is presented
        simulate(trial_duration)

        ### get the response from the PM population --> the PM neuron which is most active = selected category
        ### there are only two PM neurons --> it selects either category A or B (0 or 1)
        response_value = np.max(PM.r)
        response_idx = np.argmax(PM.r)

        ### I think, in your code, the response is only evaluated if it is high enough --> wait until it's high enough or all features have been presented
        ### I think, if all features of a fish have been presented (the presentation of the fish is complete) the model should do a response
        if response_value > response_threshold or feature_name == feature_order[-1]:
            ### if response was made --> deactivate the input (the manuscript says, this takes 30 ms)
            IT.baseline = baseline_it
            simulate(deactivate_input_duration)

            ### check if the response is equal to the category of the given fish
            category_selected = ["A", "B"][response_idx]
            print(
                f"category_true: {category_true}; category_selected: {category_selected}; after {feature_order.index(feature_name)+1} features",
            )
            if category_selected == category_true:
                ### correct response
                ### therefore, activate PPTN, here called Reward_Layer
                Reward_Layer.baseline = 1
                ### count the consecutive correct decisions, to stop after 300 consecutive correct decisions
                consecutive_correct += 1
            else:
                ### incorrect response
                ### therefore, do not activate PPTN
                Reward_Layer.baseline = 0
                ### reset the consecutive correct decisions
                consecutive_correct = 0

            ### simulate the dopamine input by activating SNc
            SNc.firing = 1
            simulate(dopamine_input_duration)

            ### stop the dopamine presentation
            Reward_Layer.baseline = 0
            SNc.firing = 0

            ### end/break the presentation of the fish --> next fish will be presented
            ### let the calcium traces decline to zero
            simulate(calcium_trace_decline_duration)
            mon.pause()
            break
#mon_dict["pop;PFC_MTL"]=["r", "baseline_rec"]
recordings = mon.get_recordings()
recording_times = mon.get_recording_times()
myx=[x['IT;r'] for x in recordings]
myx1=[x['Reward_Layer;r'] for x in recordings]
myx2=[x['PFC_MTL;r'] for x in recordings ]
myx3=[x['StrD1;r'] for x in recordings ]
myx4=[x['StrD2;r'] for x in recordings ]
myx5=[x['StrThal-PFC;r'] for x in recordings ]
import numpy as np
import matplotlib.pyplot as plt
itr=[]
def Average(lst):
    return sum(lst) / len(lst)
   
for j in range(len(myx[0])):
    itr.append(myx[0][j])
    
print(Average(itr))
fig, axes=plt.subplots(2,4)
ax=axes[0,0]
ax.plot(Average(itr))
ax=axes[0,1]
ax.plot(myx[0][1])

ax=axes[0,2]
ax.plot(myx[0][2])
ax=axes[0,3]
ax.plot(myx[0][3])
ax=axes[1,0]
ax.plot(myx[0][4])
ax=axes[1,1]
ax.plot(myx[0][5])
ax=axes[1,2]
ax.plot(myx[0][6])
ax=axes[1,3]
ax.plot(myx[0][7])
my=['IT;r', 'Reward_Layer;r', 'PFC_MTL;r', 'SNc;r', 'StrD1;r', 'StrD2;r', 'StrThal-PFC;r', 'GPi-PFC;r', 'STN;r', 'GPe;r', 'VA-PFC;r', 'PM;r']
for t in my:
    xx=[x[t] for x in recordings]
    print(xx)
    c=xx[0]
    print(c[0])
    fig, axes=plt.subplots(1,2)
    ax=axes[0]
    ax.plot(c[0])
    ax=axes[1]
    ax.plot(c[1])

for x in recordings:
   # x[mon_dict]    
    plot_list = [f"{idx + 1};{pop};r;matrix" for idx, pop in enumerate(populations)]
    print(plot_list)
    plotl=[x[m] for m in my]
for l in range(len(my)):
    print(plotl[l]) 
    if l==0:
        fig, ax=plt.subplots(1,2)
        
        ax[0].plot(consecutive_correct)
        
        ax[1].plot(consecutive_correct)
    else:
        fig, axes=plt.subplots(l+1,2)
        
        h=0
        while h<l:
           ax=axes[h,0]
           ax.plot(consecutive_correct)
           ax=axes[l,1]
           ax.plot(consecutive_correct)
           h+=1
      #  ax=axes[l,0]
       # ax.plot(plotl[l])
     #  ax=axes[l,1]
      #  ax.plot(consecutive_correct)
    
#for x in recordings:
    
#plot_list = [f"{idx + 1};{pop};r;matrix" for idx, pop in enumerate(populations)]

#for idx in range(nr_fish_presentations):
#    plot_recordings(
  #      figname=f"test_plots/new_structure/fish_presentation_{idx}.png",
 #       recordings=recordings,
  #      recording_times=recording_times,
 #       chunk=0,
 #       shape=(2, 6),
  #      plan=plot_list,
  #      time_lim=recording_times.time_lims(chunk=0, period=idx),
