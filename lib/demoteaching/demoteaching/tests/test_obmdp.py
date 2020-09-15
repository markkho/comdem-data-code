import unittest

from pyrlap.domains.gridworld import GridWorld
from pyrlap.core.util import max_index
# from mdp_lib.mcts import ValueHeuristic, ForwardSearchSparseSampling
from demoteaching.mdps.obs_belief_mdp import ObserverBeliefMDP
from demoteaching.mdps.discretizedobmdp import \
    DiscretizedObserverBeliefMDPApproximation
# from ped_irl.goalobmdp_heuristic import TerminalGoalObserverBeliefHeuristic
from itertools import product

class ObservationBeliefSmallMDPTestCase(unittest.TestCase):
    def setUp(self):
        det_planners = {}
        for mdpc in product('xo', repeat=2):
            mdpc = ''.join(mdpc)

            rewards = [{'x': -1, 'o': 0}[c] for c in mdpc]
            feature_rewards = dict(zip('pc', rewards))
            feature_rewards['w'] = 0
            feature_rewards['y'] = 5
            mdp = GridWorld(
                gridworld_array=[['w', 'p', 'y'],
                                 ['w', 'c', 'w']],
                feature_rewards=feature_rewards,
                init_state=(0, 0),
                absorbing_states=[(2, 1)],
                include_intermediate_terminal=True
            )
            planner = mdp.solve(discount_rate=.99,
                                softmax_temp=.5,
                                randchoose=.1)
            det_planners[mdpc] = planner
        self.det_planners = det_planners

        sto_planners = {}
        for mdpc in 'sw':
            non_std_t_features = {'g': {
                '2forward': {'s':.7, 'w':.3}[mdpc],
                'forward': 1 - {'s':.7, 'w':.3}[mdpc]
            }}
            mdp = GridWorld(
                gridworld_array=[['w', 'y', 'w'],
                                 ['w', 'w', 'r'],
                                 ['r', 'w', 'w'],
                                 ['g', 'w', 'g'],
                                 ['w', 'w', 'w']],
                init_state=(1, 0),
                feature_rewards={'w': 0, 'r': -1, 'g': 0, 'y': 5},
                absorbing_states=[(1, 4),],
                include_intermediate_terminal=True,
                non_std_t_features=non_std_t_features
            )
            planner = mdp.solve(discount_rate=.99,
                                softmax_temp=.5,
                                randchoose=.1)
            sto_planners[mdpc] = planner
        self.sto_planners = sto_planners


    def test_belief_state_transition(self):
        obmdp = ObserverBeliefMDP(self.det_planners,
                                  true_planner_name='oo',
                                  belief_reward=5,
                                  belief_reward_type='true_gain')
        traj = list('>^v^v^>%%')
        policy = lambda s: traj.pop(0)
        traj = obmdp.run_policy(policy)

        oo_i = obmdp.planner_order.index('oo')
        oo_ismax = max_index(traj[-1][0][0]) == oo_i
        self.assertTrue(oo_ismax)

        total_r = sum([r for s, a, ns, r in traj])
        self.assertTrue(total_r > 5)

    def test_discretizedobmdp_simple_gridworld(self):
        seed_trajs = []
        for planner in self.det_planners.values():
            for _ in range(20):
                traj = planner.run(softmax_temp=1, randchoose=.05)
                seed_trajs.append(traj)

        dobmdp = DiscretizedObserverBeliefMDPApproximation(
            planners=self.det_planners,
            true_planner_name='oo',
            belief_reward=5,
            belief_reward_type='true_gain',

            n_probability_bins=5,
            seed_trajs=seed_trajs
        )
        dobmdp.build()
        dobmdp_planner = dobmdp.solve(discount_rate=.99,
                                      softmax_temp=0.0,
                                      randchoose=0.0)
        traj = dobmdp_planner.run()

        oo_i = dobmdp.planner_order.index('oo')
        oo_ismax = max_index(traj[-1][0][0]) == oo_i
        self.assertTrue(oo_ismax)

        total_r = sum([r for s, a, ns, r in traj])
        self.assertTrue(total_r > 8.5)

    def test_discretizedobmdp_stochastic_gridworld(self):
        seed_trajs = []
        non_std_t_features = {'g': {
            '2forward': .5,
            'forward': .5
        }}
        exp_mdp = GridWorld(
            gridworld_array=[['w', 'y', 'w'],
                             ['w', 'w', 'w'],
                             ['w', 'w', 'w'],
                             ['g', 'w', 'g'],
                             ['w', 'w', 'w']],
            init_state=(1, 0),
            feature_rewards={'w': 0, 'r': 0, 'g': 0, 'y': 5},
            absorbing_states=[(1, 4), ],
            include_intermediate_terminal=True,
            non_std_t_features=non_std_t_features
        )
        exp_planner = exp_mdp.solve(discount_rate=.99,
                                    randchoose=.5,
                                    softmax_temp=0.0)
        for _ in range(50):
            traj = exp_planner.run()
            seed_trajs.append(traj)

        dobmdp = DiscretizedObserverBeliefMDPApproximation(
            planners=self.sto_planners,
            true_planner_name='s',
            belief_reward=5,
            belief_reward_type='true_gain',
            n_probability_bins=5,
            seed_trajs=seed_trajs
        )
        dobmdp_planner = dobmdp.solve(discount_rate=.99,
                                      softmax_temp=0.0,
                                      randchoose=0.0)
        traj = dobmdp_planner.run()

        s_i = dobmdp.planner_order.index('s')
        s_ismax = max_index(traj[-1][0][0]) == s_i
        self.assertTrue(s_ismax)


if __name__ == '__main__':
    unittest.main()
