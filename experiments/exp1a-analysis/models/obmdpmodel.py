import math
from itertools import product

from pyrlap.domains.gridworld import GridWorld
from demoteaching.mdps.discretizedobmdp import \
    DiscretizedObserverBeliefMDPApproximation

class OBMDPModel(object):
    def __init__(self,
                 true_mdp_code,
                 do_discount, 
                 do_randchoose,
                 do_temp,
                 show_discount,
                 show_reward,
                 show_randchoose,
                 show_temp,
                 n_bins=8,
                 seed_trajs=None,
                 disc_tf=None, 
                 solved_planner=None):
        self.show_discount = show_discount
        self.show_randchoose = show_randchoose
        self.show_temp = show_temp
        
        #=============================#

        #if the solved planner is provided, no need to
        #compute other stuff
        if solved_planner is not None:
            self.obmdp_planner = solved_planner
            self.obmdp = solved_planner.mdp
            return
        
                
        #=============================#
        #   Build set of ground MDPs  #
        #=============================#
        danger_r = -2
        goal_reward = 10
        init_ground = (0, 2)
        goal_s = (5, 2)
        
        mdp_params = []
        feature_rewards = []
        for rs in product([0, danger_r], repeat=3):
            feature_rewards.append(dict(zip('opc', rs)))
            
        mdp_codes = []
        for fr in feature_rewards:
            rfc = ['o' if fr[f] == 0 else 'x' for f in 'opc']
            rfc = ''.join(rfc)
            mdp_codes.append(rfc)
            fr['y'] = goal_reward
            fr['.'] = 0
        
        planners = {}
        for mdpc, frewards in zip(mdp_codes, feature_rewards):
            params = {
                'gridworld_array': ['.oooo.',
                                    '.oppp.',
                                    '.opccy',
                                    '.oppc.',
                                    '.cccc.'],
                'feature_rewards': frewards,
                'absorbing_states': [goal_s, ],
                'init_state': init_ground,
                'wall_action': False,
                'step_cost': 0,
                'wait_action': False,
                'include_intermediate_terminal': True
            }
            mdp = GridWorld(**params)
            planner = mdp.solve(
                softmax_temp=do_temp, 
                randchoose=do_randchoose, 
                discount_rate=do_discount)
            planners[mdpc] = planner
            
        #===========================================#
        #   Build Observer Belief MDP and support   #
        #===========================================#
        obmdp = DiscretizedObserverBeliefMDPApproximation(
            n_probability_bins=n_bins,
            seed_trajs=seed_trajs,
            branch_steps=0,
            discretized_tf=disc_tf,
            planners=planners,
            true_planner_name=true_mdp_code,
            belief_reward_type='true_gain',
            only_belief_reward=False,
            belief_reward=show_reward,
            update_includes_intention=True)
        self.obmdp = obmdp
        self.obmdp_planner = None

    
    def get_disc_tf(self):
        return self.obmdp.get_discretized_tf()
    
    def get_planner(self):
        self.obmdp_planner = self.obmdp.solve(
            discount_rate=self.show_discount,
            max_iterations=1000)
        return self.obmdp_planner
    
    def trajectory_loglikelihood(self, wtraj):
        if self.obmdp_planner is None:
            self.get_planner()

        init_state = self.obmdp.get_init_state()
        btraj = self.obmdp._wtraj_to_btraj(wtraj, init_state)
        logl = 0
        for s, a in btraj:
            adist = self.obmdp_planner.act_dist(
                s, softmax_temp=self.show_temp, 
                randchoose=self.show_randchoose)
            logl += math.log(adist[a])
        return logl
