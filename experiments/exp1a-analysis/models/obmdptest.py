import unittest
import pandas as pd
import numpy as np

from .obmdpmodel import OBMDPModel

class MyTestCase(unittest.TestCase):
    def setUp(self):
        demon_data = pd.read_pickle('../data/exp1-demon_trajs.pd.pkl')
        self.seed_trajs = list(set([tuple(t) for t in demon_data['traj']]))

        trajs_by_condrf = {}
        for (rf, cond), block in demon_data.groupby(['rf', 'cond']):
            trajs_by_condrf[cond] = trajs_by_condrf.get(cond, {})
            trajs_by_condrf[cond][rf] = [tuple(t) for t in block['traj']]
        self.trajs_by_condrf = trajs_by_condrf

    def test_fittingdata(self):
        model_params = {
            'true_mdp_code': 'xoo',
            'do_discount': .99,
            'do_randchoose': .05,
            'do_temp': .5,
            'show_discount': .9,
            'show_reward': 5,
            'show_randchoose': .05,
            'show_temp': .1,

            'n_bins': 5,
            'seed_trajs': self.seed_trajs,
            'disc_tf': None,
            'solved_planner': None
        }
        model = OBMDPModel(**model_params)

        loglikes = {}
        for cond, rf_trajs in self.trajs_by_condrf.items():
            loglikes[cond] = {}
            for rf, trajs in rf_trajs.items():
                loglikes[cond][rf] = []
                for traj in trajs:
                    loglikes[cond][rf].append(model.trajectory_loglikelihood(traj))

        self.assertTrue(
            sum(loglikes['show']['xoo']) > sum(loglikes['show']['oxx']))

        planner = model.get_planner()

        np.random.seed(0)
        max_show_fits = []
        for _ in range(20):
            btraj = planner.run(softmax_temp=0, randchoose=0)
            traj = [(w, a) for (b, w), a, (nb, nw), r in btraj]
            fit = model.trajectory_loglikelihood(traj)
            max_show_fits.append(fit)
        med_show_fits = []
        for _ in range(20):
            btraj = planner.run(softmax_temp=.1, randchoose=.05)
            traj = [(w, a) for (b, w), a, (nb, nw), r in btraj]
            fit = model.trajectory_loglikelihood(traj)
            med_show_fits.append(fit)
        min_show_fits = []
        for _ in range(20):
            btraj = planner.run(softmax_temp=.5, randchoose=.2)
            traj = [(w, a) for (b, w), a, (nb, nw), r in btraj]
            fit = model.trajectory_loglikelihood(traj)
            min_show_fits.append(fit)
        self.assertTrue(sum(max_show_fits) > sum(med_show_fits))
        self.assertTrue(sum(med_show_fits) > sum(min_show_fits))

if __name__ == '__main__':
    unittest.main()
