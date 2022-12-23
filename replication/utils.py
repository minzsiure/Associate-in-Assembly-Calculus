import numpy as np


def overlap(a, b):
    return len(set(a) & set(b))


def capk(vec, k):
    if len(np.where(vec != 0.)[0]) > k:
        indices = np.argsort(vec)
        vec[indices[:-k]] = 0.
    vec[np.where(vec != 0.)[0]] = 1.0
    return vec


def hebbian_update(in_vec, out_vec, W, beta):
    for i in np.where(out_vec != 0.)[0]:
        for j in np.where(in_vec != 0.)[0]:
            W[i, j] *= 1. + beta
    return W


def find_feedforward_matrix_index(area_combinations, from_index, to_index):
    '''
    return the index of pair [from_index, to_index] in the list Brain.area_combinations
    '''
    for i, acomb in enumerate(area_combinations):
        if all(acomb == [from_index, to_index]):
            return i

# print(np.array(np.where(~np.eye(3, dtype=bool))).T)


def generate_labels(num_classes, n):
    '''
    Generate labels based on number of classes, each class has n labels
    '''
    labels = []
    for i in range(num_classes):
        labels += [i] * n
    return np.array(labels)


def hamming_distance(vec1, vec2):
    '''
    calculate hamming distance between the two vectors.
    vec1, vec2 both have to be binary vectors
    '''
    return np.count_nonzero(vec1 != vec2)
