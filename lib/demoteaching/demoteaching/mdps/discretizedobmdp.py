from __future__ import division
from itertools import combinations_with_replacement
import logging

import numpy as np
from sklearn.neighbors import BallTree

from .obs_belief_mdp import ObserverBeliefMDP
from pyrlap.algorithms.valueiteration import ValueIteration

logger = logging.getLogger(__name__)

class DiscretizedObserverBeliefMDPApproximation(ObserverBeliefMDP):
    def __init__(self,
                 n_probability_bins=5,
                 seed_trajs=None,
                 branch_steps=0,
                 discretized_tf=None,
                 **kwargs):
        ObserverBeliefMDP.__init__(self, **kwargs)
        self.n_pbins = n_probability_bins

        self.seed_trajs = seed_trajs
        self.branch_steps = branch_steps

        if discretized_tf is not None:
            logger.debug("Discretized Transition Function Provided")
            self.disc_tf = discretized_tf
            self.belief_points = np.array(list(set([b for b, _ in self.disc_tf.keys()])))
            bp_nbrs = BallTree(self.belief_points, leaf_size=40)
            self.bp_nbrs = bp_nbrs
        else:
            self.build()
        self._create_start_state()

    # ================================= #
    #          MDP Build Methods        #
    # ================================= #

    def build(self):
        #create belief points and build transition function
        self.belief_points = []
        self._add_grid_beliefs()
        self._add_seedtraj_beliefs()
        self.belief_points = np.array(self.belief_points)
        bp_nbrs = BallTree(self.belief_points, leaf_size=40)
        self.bp_nbrs = bp_nbrs

        self._build_transition_function()

        # create start state
        self._create_start_state()

    def _create_start_state(self):
        init_state = ObserverBeliefMDP.get_init_state(self)
        init_state = self.discretize_bstate(s=init_state)
        self.init_state = init_state

    def _add_grid_beliefs(self):
        belief_grid_points = []
        for divs in combinations_with_replacement(
                range(self.n_pbins + 1),
                r=len(self.planner_order) - 1):
            b = [0, ] + list(divs) + [self.n_pbins, ]
            b = np.ediff1d(b)
            b = b / np.sum(b)
            belief_grid_points.append(tuple(b))
        self.belief_points.extend(belief_grid_points)

    def _add_seedtraj_beliefs(self):
        visited_states = set([])
        beliefs = set([])
        for traj in self.seed_trajs:
            btraj = self._wtraj_to_btraj(wtraj=traj)
            for (b, w), _ in btraj:
                beliefs.add(b)
                visited_states.add((b, w))
        branched_states = self._branch_from_states(visited_states,
                                                   self.branch_steps)
        branched_beliefs = set([b for b, w in branched_states])

        self.belief_points.extend(list(beliefs))
        self.belief_points.extend(list(branched_beliefs))

    def _branch_from_states(self, init_states, branch_steps):
        def _branch(state, depth):
            if depth == 0:
                return [state,]
            branched_states = []
            next_states = set([])
            actions = ObserverBeliefMDP.available_actions(self, state)
            for a in actions:
                tdist = ObserverBeliefMDP.transition_dist(self, state, a)
                for ns in tdist.keys():
                    next_states.add(ns)
            for ns in next_states:
                branched_states.extend(_branch(ns, depth-1))
            return branched_states

        branched_states = []
        for s in init_states:
            branched_states.extend(_branch(s, branch_steps))
        return branched_states

    def _build_transition_function(self):
        # map actual next states to discretized next states
        true_mdp = self.get_true_mdp()

        logger.debug("Building Discrete TF "+
                     "(S = %d belief points)" % len(self.belief_points))
        next_beliefs = []
        for b in self.belief_points:
            b = tuple(b)
            for w in true_mdp.get_states():
                s = (b, w)
                for a in self.available_actions(s):
                    for ns in self.transition_dist(s, a).keys():
                        nb, nw = ns
                        next_beliefs.append(nb)

        disc_b_ind = self.bp_nbrs.query(next_beliefs, return_distance=False)

        # make it so the next belief is a mixture of closest beliefs?
        # disc_b_dist, disc_b_ind = self.bp_nbrs.query(next_beliefs, k=1)
        # nb_to_disc_nb = {}
        # for nb_i, nb in enumerate(next_beliefs):
        #     nearest_idx = disc_b_ind[nb_i]
        #     neighbor = belief_points[nearest_idx]
        #     dists = disc_b_dist[nb_i]

        nb_to_disc_nb = dict(zip(next_beliefs,
                                 self.belief_points[disc_b_ind].squeeze()))
        disc_tf = {}
        for b in self.belief_points:
            b = tuple(b)
            for w in true_mdp.get_states():
                s = (b, w)
                disc_tf[s] = {}
                for a in self.available_actions(s):
                    disc_tf[s][a] = {}
                    p_norm = 0
                    for ns, p in self.transition_dist(s, a).items():
                        nb, nw = ns
                        ns = (tuple(nb_to_disc_nb[nb]), nw)
                        disc_tf[s][a][ns] = disc_tf[s][a].get(ns, 0.0) + p
                        p_norm += p
                    tdist = {ns: p / p_norm for ns, p in disc_tf[s][a].items()}
                    disc_tf[s][a] = tdist
        self.disc_tf = disc_tf

    def discretize_bstate(self, s=None, b=None):
        if b is not None:
            dist, ind = self.bp_nbrs.query([b])
            return tuple(self.belief_points[ind[0][0]])
        elif s is not None:
            b, w = s
            b = self.discretize_bstate(b=b)
            return (b, w)

    # =====================================#
    #                                     #
    #   MDP Interface Methods             #
    #                                     #
    # =====================================#

    def get_states(self):
        return list(self.disc_tf.keys())

    def get_discretized_tf(self):
        return self.disc_tf

    def transition_reward_dist(self, s, a):
        if not hasattr(self, "disc_tf"):
            return ObserverBeliefMDP.transition_reward_dist(self, s, a)

        if s not in self.disc_tf:
            s = self.discretize_bstate(s=s)
        tdist = self.disc_tf[s][a]
        trdist = {}
        for ns, p in tdist.items():
            r = self.reward(s, a, ns)
            trdist[(ns, r)] = p
        return trdist

    def get_init_state(self):
        return self.init_state

    def solve(self, discount_rate,
              softmax_temp=0.0,
              randchoose=0.0,
              **kwargs):
        planner = ValueIteration(mdp=self,
                                 transition_function=self.disc_tf,
                                 softmax_temp=softmax_temp,
                                 randchoose=randchoose,
                                 discount_rate=discount_rate,
                                 **kwargs)
        planner.solve()
        return planner