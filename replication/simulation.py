from importlib.metadata import distributions
import numpy as np
from brain import *
from utils import *
import matplotlib.pyplot as plt
import sys
import random
import copy
import seaborn as sns


def double_project_associate(x=None, num_neurons=1000, beta=0.05,
                             k=41, connection_p=0.01):
    # initialize stimulus x
    if x is None:
        x = np.zeros(num_neurons)
        stim = np.random.permutation(num_neurons)[:k]
        x[stim] = 1.0

    # initialize brain
    brain = Brain(num_areas=4, n=num_neurons, beta=beta,
                  p=connection_p, k=k)
    # 1. create assemblies A and B to stability
    # touched_neurons = set()
    # touched_neurons_size = []
    for iiter in range(9):
        _ = brain.project(x, from_area_index=0, to_area_index=1,
                          max_iterations=1, only_once=True)
        _ = brain.project(x, from_area_index=0, to_area_index=2,
                          max_iterations=1, only_once=True)

    # 2. Project A->C
    for iiter in range(9):
        # _ = brain.project(x, from_area_index=0, to_area_index=1,
        #                   max_iterations=1, only_once=True)
        _ = brain.project(brain.areas[1].activations, from_area_index=1, to_area_index=3,
                          max_iterations=1, only_once=True)
    # Project B->C
    for iiter in range(9):
        # _ = brain.project(x, from_area_index=0, to_area_index=2,
        #                   max_iterations=1, only_once=True)
        _ = brain.project(brain.areas[2].activations, from_area_index=2, to_area_index=3,
                          max_iterations=1, only_once=True)

    # # 3. Project both A,B to C
    # for iiter in range(9):
    #     # _ = brain.project(x, from_area_index=0, to_area_index=1,
    #     #                   max_iterations=1, only_once=True)
    #     _ = brain.project(brain.areas[1].activations, from_area_index=1, to_area_index=3,
    #                       max_iterations=1, only_once=True)
    #     # _ = brain.project(x, from_area_index=0, to_area_index=2,
    #     #                   max_iterations=1, only_once=True)
    #     _ = brain.project(brain.areas[2].activations, from_area_index=2, to_area_index=3,
    #                       max_iterations=1, only_once=True)
    #     touched_neurons = touched_neurons.union(
    #         np.where(brain.areas[3].activations != 0)[0])
    #     touched_neurons_size.append(len(touched_neurons))

    # 4. make a copy of brain
    # b1, b2, then project A -> C in b1, project B -> C in b2
    # compute the overlap winners of C in b1 and C in b2
    result = []
    for iiter in range(15):
        # _ = brain.project(x, from_area_index=0, to_area_index=1,
        #                   max_iterations=1, only_once=True)
        _ = brain.project(brain.areas[1].activations, from_area_index=1, to_area_index=3,
                          max_iterations=1, only_once=True)

        # _ = brain.project(x, from_area_index=0, to_area_index=2,
        #                   max_iterations=1, only_once=True)
        _ = brain.project(brain.areas[2].activations, from_area_index=2, to_area_index=3,
                          max_iterations=1, only_once=True)

        b1 = copy.deepcopy(brain)
        b2 = copy.deepcopy(brain)
        # in copy 1, project just A
        # _ = b1.project(x, from_area_index=0, to_area_index=1,
        #                max_iterations=1, only_once=True)
        _ = b1.project(brain.areas[1].activations, from_area_index=1, to_area_index=3,
                       max_iterations=1, only_once=True)

        # in copy 2, project just B
        # _ = b2.project(x, from_area_index=0, to_area_index=2,
        #                max_iterations=1, only_once=True)
        _ = b2.project(brain.areas[2].activations, from_area_index=2, to_area_index=3,
                       max_iterations=1, only_once=True)
        o = overlap(b1.areas[3].winners(), b2.areas[3].winners())
        result.append(round(float(o)/float(k), 4))

    return result


def reciprocal_project_associate(x=None, num_neurons=1000, beta=0.05,
                                 k=41, connection_p=0.01):
    # initialize stimulus x
    if x is None:
        x = np.zeros(num_neurons)
        stim = np.random.permutation(num_neurons)[:k]
        x[stim] = 1.0

    # initialize brain
    brain = Brain(num_areas=4, n=num_neurons, beta=beta,
                  p=connection_p, k=k)
    # 1. reciprocal project
    for iiter in range(9):
        _ = brain.reciprocal_project(x, area1_index=0, area2_index=1, area3_index=2,
                                     max_iterations=1, only_once=True)

    # 2. Project A->C
    for iiter in range(9):
        _ = brain.project(brain.areas[1].activations, from_area_index=1, to_area_index=3,
                          max_iterations=1, only_once=True)
    # Project B->C
    for iiter in range(9):
        _ = brain.project(brain.areas[2].activations, from_area_index=2, to_area_index=3,
                          max_iterations=1, only_once=True)

    # 4. make a copy of brain
    # b1, b2, then project A -> C in b1, project B -> C in b2
    # compute the overlap winners of C in b1 and C in b2
    result = []
    for iiter in range(15):
        _ = brain.project(brain.areas[1].activations, from_area_index=1, to_area_index=3,
                          max_iterations=1, only_once=True)

        _ = brain.project(brain.areas[2].activations, from_area_index=2, to_area_index=3,
                          max_iterations=1, only_once=True)

        b1 = copy.deepcopy(brain)
        b2 = copy.deepcopy(brain)
        # in copy 1, project just A
        _ = b1.project(brain.areas[1].activations, from_area_index=1, to_area_index=3,
                       max_iterations=1, only_once=True)

        # in copy 2, project just B
        _ = b2.project(brain.areas[2].activations, from_area_index=2, to_area_index=3,
                       max_iterations=1, only_once=True)
        o = overlap(b1.areas[3].winners(), b2.areas[3].winners())
        result.append(round(float(o)/float(k), 4))

    return result


if __name__ == "__main__":

    for i in range(10):
        print('figure %i' % (i))
        # run experiment for 10 trials and plot avrage
        x = np.zeros(1000)
        stim = np.random.permutation(1000)[:41]
        x[stim] = 1.0
        concat_proj = np.zeros((10, 15))
        concat_reci = np.zeros((10, 15))

        for itrial in range(10):
            result_proj = double_project_associate(x)
            concat_proj[itrial] = result_proj

            result_reci = reciprocal_project_associate(x)
            concat_reci[itrial] = result_reci

        proj_mean = np.mean(concat_proj, axis=0)
        reci_mean = np.mean(concat_reci, axis=0)
        plt.figure()
        plt.plot([i+1 for i in range(len(proj_mean))],
                 proj_mean, label="project")
        plt.plot([i+1 for i in range(len(reci_mean))],
                 reci_mean, label="reci-project")
        plt.legend()
        plt.xlabel('time (step)')
        plt.ylabel('overlap (%)')
        # plt.show()
        plt.savefig('ten_trials_average/%i.png' % (i))

    # plot associate
    # for i in range(10):
    #     x = np.zeros(1000)
    #     stim = np.random.permutation(1000)[:41]
    #     x[stim] = 1.0
    #     result_proj = double_project_associate(x)
    #     result_reci = reciprocal_project_associate(x)
    #     plt.figure()
    #     plt.plot([i+1 for i in range(len(result_proj))],
    #              result_proj, label="project")
    #     plt.plot([i+1 for i in range(len(result_reci))],
    #              result_reci, label="reci-project")
    #     plt.legend()
    #     plt.xlabel('time (step)')
    #     plt.ylabel('overlap (%)')
    #     # plt.show()
    #     plt.savefig('reciprocal_associate_figures/%i.png' % (i))
