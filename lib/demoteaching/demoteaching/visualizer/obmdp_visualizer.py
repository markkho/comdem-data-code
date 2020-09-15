from __future__ import division
from itertools import product

import pytweening

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mdp_lib.domains.gridworld import GridWorld
from mdp_lib.domains.gridworldvis import visualize_trajectory, plot_agent_location

def initialize_grids(artist_dict,
                     mdp_codes,
                     mdp_param_dicts,
                     main_mdp_param,
                     init_belief,
                     init_state,
                     main_colspan=None,
                     main_rowspan=3,
                     main_rows=4,
                     main_topleft=(2, 1),
                     bars_aspect=1,
                     fig=None):
    if fig is None:
        fig = plt.figure(figsize=(40, 30))

    fig_cell_shapes = (2 + main_rows, len(mdp_codes))
    gridspec.GridSpec(*fig_cell_shapes)

    # plot observable
    if main_colspan is None:
        main_colspan = len(mdp_codes) - 2
    colspan = main_colspan
    rowspan = main_rowspan
    ax = plt.subplot2grid(fig_cell_shapes, main_topleft,
                          colspan=colspan, rowspan=rowspan)
    GridWorld(**main_mdp_param).plot(ax=ax)
    artist_dict['main_plot'] = ax
    main_agent = plot_agent_location(state=init_state,
                                     ax=ax)
    artist_dict['main_agent'] = main_agent

    # plot possible MDPs
    for mdpi, mdpc in enumerate(mdp_codes):
        ax = plt.subplot2grid(fig_cell_shapes, (0, mdpi))
        GridWorld(**mdp_param_dicts[mdpc]).plot(ax=ax)
        artist_dict['subplots'][mdpc] = ax
        sp_agent = plot_agent_location(state=init_state,
                                       ax=ax)
        artist_dict['subplot_agents'][mdpc] = sp_agent

    # plot beliefs
    init_belief = dict(zip(mdp_codes, init_belief))
    for mdpi, mdpc in enumerate(mdp_codes):
        ax = plt.subplot2grid(fig_cell_shapes, (1, mdpi))
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_yticks([0, .5, 1])
        ax.tick_params(axis='y', labelsize=30)
        ax.set_aspect(bars_aspect)

        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if mdpi != 0:
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)

        artist_dict['beliefplots'][mdpc] = ax
        artist_dict['beliefbars'][mdpc] = ax.bar(.5, init_belief[mdpc],
                                                 color='blue',
                                                 width=.4)[0]

    # frame1.axes.get_yaxis().set_visible(False)
    fig.tight_layout()

def draw_agent_step(artist_dict, mdp_codes, traj, traj_step,
                    interpolation=0):
    wtraj = [(w, a) for (b, w), a in traj]

    state = wtraj[traj_step][0]

    if traj_step < (len(traj) - 1):
        next_state = wtraj[traj_step + 1][0]
    else:
        next_state = None

    # Plot in main plot
    ax = artist_dict['main_plot']
    main_agent = artist_dict['main_agent']
    plot_agent_location(state=state,
                        next_state=next_state,
                        agent=main_agent,
                        interpolation=interpolation,
                        ax=ax)

    # plot subplots
    for mdpc in mdp_codes:
        # update agents
        ax = artist_dict['subplots'][mdpc]
        sp_agent = artist_dict['subplot_agents'][mdpc]
        plot_agent_location(state=state,
                            next_state=next_state,
                            agent=sp_agent,
                            interpolation=interpolation,
                            ax=ax)

def draw_agent_trajpath(artist_dict, mdp_codes, traj, traj_step,
                        interpolation=0):
    wtraj = [(w, a) for (b, w), a in traj]

    # Plot in main plot
    ax = artist_dict['main_plot']
    if artist_dict['main_path'] is not None:
        traj_artists = artist_dict['main_path']
        for p in traj_artists:
            p.remove()

    traj_artists = visualize_trajectory(
        traj=wtraj[:traj_step + 1],
        axis=ax, lw=15,
        color='g', outline=True,
        jitter_var=0
    )
    all_paths = []
    all_paths.extend(traj_artists['outline_patches'])
    all_paths.extend(traj_artists['traj_patches'])
    all_paths.extend(traj_artists['action_patches'])
    artist_dict['main_path'] = all_paths

    # plot subplots
    for mdpc in mdp_codes:
        # update agents
        ax = artist_dict['subplots'][mdpc]

        if mdpc in artist_dict['subplot_paths']:
            traj_artists = artist_dict['subplot_paths'][mdpc]
            for p in traj_artists:
                p.remove()

        traj_artists = visualize_trajectory(
            traj=wtraj[:traj_step + 1],
            axis=ax, lw=5,
            color='g', outline=True,
            jitter_var=0
        )
        all_paths = []
        all_paths.extend(traj_artists['outline_patches'])
        all_paths.extend(traj_artists['traj_patches'])
        all_paths.extend(traj_artists['action_patches'])
        artist_dict['subplot_paths'][mdpc] = all_paths

def draw_beliefs(artist_dict, mdp_codes, traj, traj_step):
    (b, _), _ = traj[traj_step]
    b = dict(zip(mdp_codes, b))

    # plot subplots
    for mdpc in mdp_codes:
        ax = artist_dict['subplots'][mdpc]
        ax = artist_dict['beliefplots'][mdpc]
        # if mdpc in artist_dict['beliefbars']:
        #     artist_dict['beliefbars'][mdpc].remove()
        # artist_dict['beliefbars'][mdpc] = ax.bar(.5, b[mdpc],
        #                                          color='blue',
        #                                          width=.4)[0]
        artist_dict['beliefbars'][mdpc].set_height(b[mdpc])

def animate_belief_gridworld_trajectory(
        traj,
        mdp_codes,
        mdp_param_dicts,
        main_mdp_param,
        filename,

        main_colspan=None,
        main_rowspan=3,
        main_rows=4,
        main_topleft=(2,1),
        bars_aspect=1,

        move_interval=1000,
        interval_frames=1,
        traj_responsetimes=None,
        fig=None,
        plot_traj_path=True,
        plot_agent=True
    ):
    if fig is None:
        fig = plt.figure(figsize=(40, 30))

    artist_dict = {
        'main_plot': None,
        'subplots': {},
        'beliefplots': {},
        'beliefbars': {},
        'subplot_agents': {},
        'subplot_paths': {},
        'main_agent': None,
        'main_path': None
    }

    def animate(step):
        if step is None:
            return []

        traj_step, interval_frame = step

        updated_artists = []
        if plot_agent:
            draw_agent_step(artist_dict, mdp_codes, traj, traj_step,
                            interpolation=interval_frame/interval_frames)
            updated_artists.append(artist_dict['main_agent'])
            updated_artists.extend(artist_dict['subplot_agents'].values())

        if plot_traj_path and interval_frame != 0:
            draw_agent_trajpath(
                artist_dict, mdp_codes, traj,
                traj_step,
                interpolation=interval_frame/interval_frames)
            all_traj_paths = []
            for traj_paths in artist_dict['subplot_paths'].values():
                all_traj_paths.extend(traj_paths)
            updated_artists.extend(all_traj_paths)

        draw_beliefs(artist_dict, mdp_codes, traj, traj_step)
        updated_artists.extend(artist_dict['beliefbars'].values())

        return updated_artists
        
    def init():
        b, w = traj[0][0]
        initialize_grids(
            artist_dict=artist_dict,
            mdp_codes=mdp_codes,
            mdp_param_dicts=mdp_param_dicts,
            main_mdp_param=main_mdp_param,
            init_belief=b,
            init_state=w,
            fig=fig,

            main_colspan=main_colspan,
            main_rowspan=main_rowspan,
            main_rows=main_rows,
            main_topleft=main_topleft,
            bars_aspect=bars_aspect
        )
        artist_list = [artist_dict['main_agent']]
        artist_list.extend(artist_dict['subplot_agents'].values())
        artist_list.extend(artist_dict['beliefbars'].values())
        return artist_list

    if traj_responsetimes is None:
        traj_responsetimes = [0,]*len(traj)

    interval = move_interval/interval_frames

    frames = []
    for step_i, frame_i in product(range(len(traj)), range(interval_frames)):
        frames.append((step_i, frame_i))
        if frame_i == 0:
            n_wait_frames = int(traj_responsetimes[step_i]/interval)
            frames.extend([None,]*n_wait_frames)

    ani = animation.FuncAnimation(fig=fig,
                                  func=animate,
                                  init_func=init,
                                  frames=frames,
                                  interval=interval,
                                  blit=True)
    ani.save(filename)