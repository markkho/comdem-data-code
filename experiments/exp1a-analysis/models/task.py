from itertools import product

from mdp_lib.domains.gridworld import GridWorld

goal_reward=50
true_belief_reward = 50
danger_reward = -10
step_cost = 0
wall_action = False
wait_action = False
init_ground_state=(0, 2)
ground_goal_state=(5, 2)

base_discount_rate = .99
base_softmax_temp = 1
obmdp_discount_rate = .99
# true_mdp_code='oox'

obmdp_softmax_temp=1

belief_reward_isterminal = False

seed_trajs=None
discretized_tf=None

state_features = [
            '.oooo.',
            '.oppp.',
            '.opccy',
            '.oppc.',
            '.cccc.'
        ]

#=============================#
#   Build set of ground MDPs  #
#=============================#
mdp_params = []
feature_rewards = [dict(zip('opc', rs)) for rs in product([0, danger_reward],
                                                          repeat=3)]
mdp_codes = []
for frewards in feature_rewards:
    rfc = ['o' if frewards[f] == 0 else 'x' for f in 'opc']
    rfc = ''.join(rfc)
    mdp_codes.append(rfc)
    frewards['y'] = goal_reward
    frewards['.'] = 0

for mdpc, frewards in zip(mdp_codes, feature_rewards):
    params = {
        'gridworld_array': state_features,
        'feature_rewards': frewards,
        'absorbing_states': [ground_goal_state, ],
        'init_state': init_ground_state,
        'wall_action': wall_action,
        'step_cost': step_cost,
        'wait_action': wait_action,
        'discount_rate': base_discount_rate
    }
    mdp_params.append(params)

#===========================================#
#   Build Observer Belief MDP and support   #
#===========================================#
ob_mdp_params = {
    'init_ground_state': init_ground_state,
    'mdp_params': mdp_params,
    'mdp_codes': mdp_codes,
    'MDP': GridWorld,
    'base_softmax_temp': base_softmax_temp,
    'true_belief_reward': true_belief_reward,
    'base_policy_type': 'softmax',
    'true_mdp_i': None,
    'belief_reward_isterminal': False,

    'discount_rate': obmdp_discount_rate,

    'discretized_tf': discretized_tf
}