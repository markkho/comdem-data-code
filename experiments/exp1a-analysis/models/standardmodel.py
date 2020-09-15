import math, time
import numpy as np
from pyrlap.domains.gridworld import GridWorld


class StandardPlanningModel(object):
    def __init__(self, true_mdp_code, do_discount, do_randchoose, do_temp):
        danger_r = -2
        goal_reward = 10
        init_ground = (0, 2)
        goal_s = (5, 2)

        fr_vals = [{'x': danger_r, 'o': 0}[v] for v in true_mdp_code]
        feature_rewards = dict(zip('opc', fr_vals))
        feature_rewards['y'] = goal_reward
        params = {
            'gridworld_array': ['.oooo.',
                                '.oppp.',
                                '.opccy',
                                '.oppc.',
                                '.cccc.'],
            'feature_rewards': feature_rewards,
            'absorbing_states': [goal_s, ],
            'init_state': init_ground,
            'wall_action': False,
            'step_cost': 0,
            'wait_action': False,
        }
        self.gw = GridWorld(**params)
        self.planner = self.gw.solve(
            discount_rate=do_discount,
            softmax_temp=do_temp,
            randchoose=do_randchoose)

    def trajectory_loglikelihood(self, wtraj):
        logl = 0
        for s, a in wtraj:
            adist = self.planner.act_dist(s)
            logl += math.log(adist[a])
        return logl