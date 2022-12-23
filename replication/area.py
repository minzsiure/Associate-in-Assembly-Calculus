import numpy as np


class Area:
    '''
    An Area contains a group of neurons.
    They may have pairwise recurrent connections.
    '''

    def __init__(self, n=1000, p=None, k=None):
        # init n, number of neurons
        self.n = n

        # init k, number of winners
        self.k = k

        # init p, sparcity
        self.p = p

        # neurons are inhibited to do recurrent/feedforward
        # activation unless disinhibited
        self.inhibited = True

        # init recurrent connections
        self.recurrent_connections = self.sample_initial_connections()

        # init y, the assembly
        self.activations = np.zeros(self.n, dtype=float)

    def winners(self):
        """
        when access,
        return index of winners in self.activations
        """
        return np.nonzero(self.activations)[0]

    def disinhibit(self):
        self.inhibited = False

    def inhibit(self):
        self.inhibited = True

    def sample_initial_connections(self):
        '''
        draw recurrenct connections based on sparcity value p
        '''
        connections = np.random.binomial(
            1, self.p, size=(self.n, self.n)).astype("float64")
        # no self loop
        np.fill_diagonal(connections, 0)
        return connections

    def wipe_activations(self):
        '''
        Reset all activation values y to be 0s
        '''
        self.activation = np.zeros(self.n)
