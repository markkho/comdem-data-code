#!/usr/bin/env python

import numpy as np
from scipy.special import softmax

from pyrlap.core.mdp import MDP as MDPClass
from pyrlap.core.agent import Agent

'''
An observer belief MDP takes a set of MDPs and tracks a belief state over them
that can be used by a showing agent.
'''

class ObserverBeliefMDP(MDPClass):
    def __init__(self,
                 planners : "Expects a a dictionary of base planners that have"
                            "an intermediate terminal state for calculating"
                            "the terminal belief reward (if that's set).",
                 true_planner_name,
                 belief_reward_type : "Options include 'true_gain' (the amount"
                                      "that the belief on the true mdp "
                                      "increases each timestep), 'max_diff' "
                                      "(the change in the maximum difference"
                                      "each timestep), and 'terminal' (the amount"
                                      "of belief on the true mdp by the end of "
                                      "the task.'" = 'true_gain',
                 only_belief_reward=False,
                 belief_reward=None,
                 update_includes_intention=True):
        '''
        Handles observer belief-MDP
        '''

        self.true_planner = planners[true_planner_name]
        self.planner_order = sorted(list(planners.keys()))
        self.true_planner_i = self.planner_order.index(true_planner_name)
        self.planners = [planners[pname] for pname in self.planner_order]

        self.belief_reward_type = belief_reward_type
        if belief_reward is None:
            belief_reward = 0
        self.belief_reward = belief_reward
        self.only_belief_reward = only_belief_reward
        self.update_includes_intention = update_includes_intention

        self.init_ground_state = self.true_planner.mdp.get_init_state()

    def get_init_state(self):
        b = tuple(np.ones(len(self.planners))/len(self.planners))
        w = self.init_ground_state
        return (b, w)

    def _calc_reward(self, s, a, ns):
        b, w = s
        nb, nw = ns

        wr = self.true_planner.mdp.reward(s=w, a=a, ns=nw)

        if self.belief_reward_type == 'true_gain':
            bchange = nb[self.true_planner_i] - b[self.true_planner_i]
            br = bchange * self.belief_reward
        elif self.belief_reward_type == 'max_diff':
            bmax_nontrue = 0
            nbmax_nontrue = 0
            for bi, prob, nprob in enumerate(zip(b, nb)):
                if bi == self.true_planner_i:
                    continue
                bmax_nontrue = max(bmax_nontrue, prob)
                nbmax_nontrue = max(nbmax_nontrue, nprob)
            bmax_diff = b[self.true_planner_i] - bmax_nontrue
            nbmax_diff = nb[self.true_planner_i] - nbmax_nontrue
            diff_change = nbmax_diff - bmax_diff
            br = diff_change * self.belief_reward
        elif self.belief_reward_type == 'terminal':
            if w == self.true_planner.mdp.intermediate_terminal:
                br = self.belief_reward * nb[self.true_planner_i]
            else:
                br = 0

        r = br
        if not self.only_belief_reward:
            r += wr
        return r

    def reward(self, s, a, ns):
        return self._calc_reward(s, a, ns)

    def transition_reward_dist(self, s, a):
        b, w = s

        # what is the probability of each planner taking action a?
        # i.e. the action likelihood
        a_lhood = np.array([pol.act_dist(w)[a] for pol in self.planners])
        nsr_dist = {} #next belief-world state, reward distribution

        # iterate over all possible next world and reward states in the
        # true planner's mdp
        nwr_p = self.true_planner.mdp.transition_reward_dist(w, a)
        for (nw, wr), p in nwr_p.items():

            # calculate the belief transition
            if p == 0:
                continue
            # what is the probability of transitioning to nw under each possible
            # planner? i.e. the transition likelihood term
            nw_lhood = []
            for pl in self.planners:
                tprob = pl.mdp.transition_dist(w, a).get(nw, 0)
                nw_lhood.append(tprob)
            nw_lhood = np.array(nw_lhood)

            if self.update_includes_intention:
                nb = np.log(a_lhood)+ np.log(nw_lhood) + np.log(b)
            else:
                nb = np.log(nw_lhood) + np.log(b)
            nb = tuple(softmax(nb))
            # nb = tuple(nb / np.sum(nb))

            # calculate belief reward
            r = self._calc_reward(s, a, (nb, nw))

            nsr_dist[((nb, nw), r)] = p
        return nsr_dist

    def is_terminal(self, state):
        b, w = state
        return self.true_planner.mdp.is_terminal(w)

    def available_actions(self, s):
        b, w = s
        return self.true_planner.mdp.available_actions(w)

    def get_true_mdp(self):
        return self.true_planner.mdp

    def _wtraj_to_btraj(self, wtraj, init_state=None):
        # if true_obmdp is None:
        #     true_obmdp = super(DiscretizedObserverBeliefMDPApproximation, self)
        if init_state is None:
            init_state = ObserverBeliefMDP.get_init_state(self)

        btraj = []
        s = init_state
        for ti in range(len(wtraj)):
            a = wtraj[ti][1]
            btraj.append((s, a))
            if ti == (len(wtraj) - 1):
                break
            pnw = wtraj[ti + 1][0]
            tdist = self.transition_dist(s, a)
            for nb, nw in tdist.keys():
                if nw == pnw:
                    break
            s = (nb, nw)
        return btraj

class ObserverBeliefMDPGroundPolicyWrapper(Agent):
    def __init__(self, ground_policy):
        self.ground_policy = ground_policy

    def __call__(self, state):
        return self.ground_policy.get_action(state[1])

    def dist(self, state):
        return self.ground_policy