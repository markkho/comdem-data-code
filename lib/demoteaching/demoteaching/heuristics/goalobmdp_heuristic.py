import numpy as np

from mdp_lib.mcts import ValueHeuristic


#==========================================#
#                                          #
#    Heuristic for OBMDP in a gridworld    #
#                                          #
#==========================================#
class TerminalGoalObserverBeliefHeuristic(ValueHeuristic):
    #todo use a policy to speed up _vmin, _vmax calculations
    #todo take into account returning to a w state with a worse b state
    #todo vmin/vmax at higher depths can be bounded if lower ones are known
    def __init__(self,
                 true_belief_reward=None,
                 goal_reward=None,
                 rmin=None,
                 discount_rate=None,
                 exp_steps_to_goal=None,
                 min_steps_to_goal=None,
                 true_mdp_i=None,
                 belief_reward_isterminal=True,
                 max_horizon=None
                 ):

        if discount_rate == 1 and max_horizon is None:
            raise ValueError("Without a discount or finite horizon, vmin/vmax is not defined")

        self.rmax = goal_reward + true_belief_reward
        self.rmin = rmin
        self._vmax = goal_reward + true_belief_reward
        if discount_rate < 1:
            self._vmin = rmin / (1 - discount_rate)
        else:
            self._vmin = rmin*max_horizon
        self.discount_rate = discount_rate
        self.goal_reward = goal_reward
        self.exp_steps_to_goal = exp_steps_to_goal
        self.min_steps_to_goal = min_steps_to_goal
        self.true_mdp_i = true_mdp_i
        self.true_belief_reward = true_belief_reward
        self.max_horizon = max_horizon

        self.steps_to_vmin = {}
        self.belief_reward_isterminal = belief_reward_isterminal
        self.ob_mdp = None

    def __hash__(self):
        return hash((
            self.rmax,
            self.rmin,
            self.discount_rate,
            self._vmax,
            self._vmin,
            self.belief_reward_isterminal
        ))

    def set_obmdp(self, ob_mdp):
        self.ob_mdp = ob_mdp
        self.belief_reward_isterminal = ob_mdp.belief_reward_isterminal
        self.true_mdp_i = ob_mdp.true_mdp_i

    def vmax(self, s=None, d=None):
        """
        You can't do better than getting the full reward (goal + 1.0 belief),
        discounted by the distance to the goal.
        """
        if s is None:
            return self._vmax

        b, w = s
        steps = self.min_steps_to_goal(s[1])
        if self.belief_reward_isterminal:
            return self._vmax*self.discount_rate**(steps-1)
        else:
            max_br = (1 - b[self.true_mdp_i])*self.true_belief_reward
            return max_br + self.goal_reward*self.discount_rate**(steps-1)

    def vmin(self, s=None, d=None):
        """
        You can't do worse than getting rmin every step and then entering
        the goal (but not getting any reward for beliefs).
        """
        if self.exp_steps_to_goal is None:
            return self._vmin

        steps = self.exp_steps_to_goal(s[1])
        if steps not in self.steps_to_vmin:
            discounts = np.ones(steps)*self.discount_rate
            pow = np.arange(steps)
            max_danger = np.sum(np.power(discounts, pow)*self.rmin)
            vmin = max_danger + self.goal_reward*(self.discount_rate**steps)
            self.steps_to_vmin[steps] = vmin

        if self.belief_reward_isterminal:
            return self.steps_to_vmin[steps]
        else:
            b, w = s
            min_br = -b[self.true_mdp_i] * self.true_belief_reward
            return min_br + self.steps_to_vmin[steps]