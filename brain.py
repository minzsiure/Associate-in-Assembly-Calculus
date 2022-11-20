import numpy as np
import heapq
from collections import defaultdict
from scipy.stats import binom
from scipy.stats import truncnorm
import math
import random


class Stimulus:
    def __init__(self, k):
        self.k = k


class Area:
    def __init__(self, name, n, k, beta=0.05):
        self.name = name
        self.n = n
        self.k = k

        # default
        self.beta = beta

        # betas from stimuli into area
        self.stimulus_beta = {}

        # betas from areas into area
        self.area_beta = {}

        self.w = 0
        # list of winners currently
        self.winners = []
        self.new_w = 0

        # new winners computed during projection
        # DO NOT use outside of internal projection function
        self.new_winners = []

        # all winners in each round
        self.saved_winners = []

        # list of size of support in each round
        self.saved_w = []
        self.num_first_winners = -1

        # whether to fix/freeze the assembly (winners) in this area
        self.fixed_assembly = False

        # whether to fully stimulate this area
        self.explicit = False

    def update_winners(self):
        '''
        Update winners
        '''
        self.winner = self.new_winners  # update winners
        if not self.explicit:  # not fully stimulate
            self.w = self.new_w

    def update_stimulus_beta(self, name, new_beta):
        '''
        Given a stimulus name and a beta value,
        update the beta corresponding to this stimulus accordingly.
        '''
        self.stimulus_beta[name] = new_beta

    def update_area_beta(self, name, new_beta):
        '''
        Given an area name and a beta value,
        update the beta corresponding to this area accordingly.
        '''
        self.area_beta[name] = new_beta

    def fix_assembly(self):
        '''
        Set `fixed_assembly` to True
        '''
        if not self.winners:
            raise ValueError(
                'Area %s does not have assembly; cannot fix.' % self.name)
            return
        self.fixed_assembly = True

    def unfix_assembly(self):
        self.fixed_assembly = False
