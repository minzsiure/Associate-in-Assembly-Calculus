import brain
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt
import brain_util as bu
import sys

from collections import OrderedDict


def association_grand_sim(n=100000, k=317, p=0.01, beta=0.05, min_iter=10, max_iter=20):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_stimulus("stimB", k)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)

    ###############################################
    # project stimulus to area
    # NO projection from area to area
    print('stimA --> A, stimB --> B')
    print('create assemblies A, B to stability')
    b.project({"stimA": ["A"], "stimB": ["B"]}, {})
    for i in range(9):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A"], "B": ["B"]})
    print()
    ###############################################
    ###############################################
    print('project A --> C')
    b.project({"stimA": ["A"]}, {"A": ["A", "C"]})
    for i in range(9):
        b.project({"stimA": ["A"]},
                  {"A": ["A", "C"], "C": ["C"]})
    print()
    print('check A --> B', b.connectomes['A']['B'].shape, np.count_nonzero(
        b.connectomes['A']['B']), np.max(b.connectomes['A']['B']))
    print('check B --> A', b.connectomes['B']['A'].shape, np.count_nonzero(
        b.connectomes['B']['A']), np.max(b.connectomes['B']['A']))

    print('project B --> C')
    b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
    for i in range(9):
        b.project({"stimB": ["B"]},
                  {"B": ["B", "C"], "C": ["C"]})
    print()
    ###############################################
    ############### CO-PROJECTION ####################
    print('project A,B --> C')
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"]})
    for i in range(min_iter-2):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]})
    print()
    ###############################################
    ################ CHECK OVERLAP ###################
    results = {}
    for i in range(min_iter, max_iter+1):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]})
        b_copy1 = copy.deepcopy(b)
        b_copy2 = copy.deepcopy(b)
        # in copy 1, project just A

        b_copy1.project({"stimA": ["A"]}, {})
        b_copy1.project({}, {"A": ["C"]})

        # in copy 2, project just B
        b_copy2.project({"stimB": ["B"]}, {})
        b_copy2.project({}, {"B": ["C"]})
        o = bu.overlap(b_copy1.areas["C"].winners,
                       b_copy2.areas["C"].winners)
        results[i] = float(o)/float(k)
        print()
    return results


def memory_transfer(n=100000, k=317, p=0.01, beta=0.05, min_iter=10, max_iter=20):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    overlap = []

    ###############################################
    # project stimulus to area
    # NO projection from area to area
    print('stimA --> A, stimA --> B')
    # print('create assemblies A, B to stability')
    b.project({"stimA": ["A", "B"]}, {})

    # Create assemblies A and B to stability
    for i in range(9):
        print("A.w=" + str(b.areas["A"].w))
        print("B.w=" + str(b.areas["B"].w))
        b.project({"stimA": ["A", "B"]},
                  {"A": ["A"], "B": ["B"]})
        o = bu.overlap(b.areas["B"].winners, b.areas["A"].winners)
        overlap.append(float(o)/float(k))
        print('overlap', overlap)

    # Coprojection, A --> B, B --> A
    print('co-projection')
    for i in range(1, 9):
        print("A.w=" + str(b.areas["A"].w))
        print("B.w=" + str(b.areas["B"].w))
        b.project({"stimA": ["A", "B"]},
                  {"A": ["A", "B"], "B": ["B", "A"]})
        o = bu.overlap(b.areas["B"].winners, b.areas["A"].winners)
        overlap.append(float(o)/float(k))
        print('overlap', overlap)
    print()

    # for i in range(min_iter, max_iter+1):
    #     b.project({"stimA": ["A", "B"]},
    #               {"A": ["A", "B"], "B": ["B", "A"]})
    #     b_copy1 = copy.deepcopy(b)
    #     b_copy2 = copy.deepcopy(b)

    #     # in copy 1, project just A
    #     print('project just A')
    #     b_copy1.project({"stimA": ["A"]}, {})
    #     b_copy1.project({}, {"A": ["B"]})

    #     # in copy 2, project just B
    #     print('project just B')
    #     b_copy2.project({"stimA": ["B"]}, {})
    #     b_copy2.project({}, {"B": ["A"]})

    #     o = bu.overlap(b_copy1.areas["B"].winners, b_copy2.areas["A"].winners)
    #     overlap = float(o)/float(k)

    #     results[i] = overlap
    #     print('overlap', overlap)
    #     print()
    return overlap


def memory_transfer_2(n=100000, k=317, p=0.01, beta=0.05, min_iter=10, max_iter=20):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)

    ###############################################
    # project stimulus to area
    # NO projection from area to area
    print('stimA --> A', 'stimA --> B')
    # print('create assemblies A, B to stability')
    b.project({"stimA": ["A", "B"]}, {})

    # Create assemblies A to stability
    b.areas["B"].fix_assembly()
    for i in range(10):
        b.project({"stimA": ["A", "B"]},
                  {"A": ["A"]})
        print("A.w=" + str(b.areas["A"].w))
    b.areas["B"].unfix_assembly()

    # Freeze assembly in A and start projecting A <-> B
    b.areas["A"].fix_assembly()
    b.project({}, {"A": ["B"]})
    for i in range(10):
        b.project({}, {"A": ["B"], "B": ["A", "B"]})
        print("B.w=" + str(b.areas["B"].w))
    b.areas["A"].unfix_assembly()

    # Coprojection, A --> B, B --> A
    results = {}
    overlap = float(bu.overlap(b.areas["A"].winners,
                               b.areas["B"].winners))/float(k)
    results[0] = overlap
    print("overlap", overlap)

    print('co-projection')
    b.project({"stimA": ["A", "B"]},
              {"A": ["A", "B"], "B": ["B", "A"]})

    for i in range(1, 9):
        b.project({"stimA": ["A", "B"]},
                  {"A": ["A", "B"], "B": ["B", "A"]})
        o = bu.overlap(b.areas["A"].winners, b.areas["B"].winners)
        overlap = float(o)/float(k)
        results[i] = overlap
        print("overlap", overlap)
        print()
    return results


def fixed_assembly_recip_proj(n=100000, k=317, p=0.01, beta=0.05):
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)

    # Will project fixes A into B
    b.add_area("B", n, k, beta)
    b.project({"stimA": ["A"]}, {})
    print("A.w=" + str(b.areas["A"].w))
    for i in range(20):
        b.project({"stimA": ["A"]}, {"A": ["A"]})
        print("A.w=" + str(b.areas["A"].w))

    # Freeze assembly in A and start projecting A <-> B
    b.areas["A"].fix_assembly()
    b.project({}, {"A": ["B"]})
    for i in range(20):
        b.project({}, {"A": ["B"], "B": ["A", "B"]})
        print("B.w=" + str(b.areas["B"].w))

    # If B has stabilized, this implies that the A->B direction is stable.
    # Therefore to test that this "worked" we should check that B->A restores A
    print("Before B->A, A.w=" + str(b.areas["A"].w))
    b.areas["A"].unfix_assembly()
    b.project({}, {"B": ["A"]})
    print("After B->A, A.w=" + str(b.areas["A"].w))
    for i in range(20):
        b.project({}, {"B": ["A"], "A": ["A"]})
        print("A.w=" + str(b.areas["A"].w))
    overlaps = bu.get_overlaps(
        b.areas["A"].saved_winners[-22:], 0, percentage=True)
    print(overlaps)


if __name__ == '__main__':
    # np.set_printoptions(threshold=sys.maxsize)
    b = memory_transfer()
    print(b)
    # b = fixed_assembly_recip_proj()
