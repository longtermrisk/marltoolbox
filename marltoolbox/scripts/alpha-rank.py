import matplotlib.pyplot as plt

# import feasible_set_figure as game_utils
# import nashpy as nash
import numpy as np
from open_spiel.python.algorithms import projected_replicator_dynamics
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.egt import alpharank

# import copy

# import prd
# from scipy.special import expit
# import pdb

SMALL_META_PAYOUT_1 = np.array(
    [[4.0, 0.0, 4.0], [0.0, 2.0, 2.0], [4.0, 2.0, 3.0]]
)
SMALL_META_PAYOUT_2 = np.array(
    [[1.0, 0.0, 1.0], [0.0, 2.0, 2.0], [1.0, 2.0, 1.5]]
)

# Util, Egal, Mix, [Util, Egal], [Util, Mix] [Egal, Mix], [All]
# LARGE_META_PAYOUT_1 = np.array(
#     [
#         [4.0, 0.0, 2.0, 4.0, 4.0, 2.0, 4.0],
#         [0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0],
#         [2.0, 1.0, 3.0, 1.0, 3.0, 3.0, 3.0],
#         [4.0, 2.0, 3.0, 1.0, 3.0, 3.0, 3.0],
#     ]
# )
# LARGE_META_PAYOUT_2 = np.array(
#     [[1.0, 0.0, 1.0], [0.0, 2.0, 2.0], [1.0, 2.0, 1.5]]
# )
MIS = 0.0
# Util, Egal, Mix, [Util, Egal], [Util, Mix] [Egal, Mix], [All]
META_PAYOUT_1 = np.array(
    [
        [
            4.0,
            MIS,
            MIS,
            4.0,
            4.0,
            MIS,
            4.0,
        ],
        [
            MIS,
            2.0,
            MIS,
            2.0,
            MIS,
            2.0,
            2.0,
        ],
        [
            MIS,
            MIS,
            3.0,
            MIS,
            3.0,
            3.0,
            3.0,
        ],
        [4.0, 2.0, MIS, 3.0, 4.0, 2.0, 3.0],
        [4.0, MIS, 3.0, 4.0, 3.5, 3.0, 3.5],
        [MIS, 2.0, 3.0, 2.0, 3.0, 2.5, 2.5],
        [4.0, 2.0, 3.0, 3.0, 3.5, 2.5, 3.0],
    ]
)
META_PAYOUT_2 = np.array(
    [
        [1.0, MIS, MIS, 1.0, 1, MIS, 1.0],
        [
            MIS,
            2.0,
            MIS,
            2.0,
            MIS,
            2.0,
            2.0,
        ],
        [
            MIS,
            MIS,
            1.5,
            MIS,
            1.5,
            1.5,
            1.5,
        ],
        [1.0, 2.0, MIS, 1.5, 1.0, 2.0, 1.5],
        [1.0, MIS, 1.5, 1.0, 1.25, 1.5, 1.75],
        [MIS, 2.0, 1.5, 2.0, 1.5, 1.75, 1.75],
        [1.0, 2.0, 1.5, 1.5, 1.25, 1.75, 1.5],
    ]
)

# "(OrderedSet(['egalitarian', 'mixed', 'utilitarian']),
# OrderedSet(['egalitarian', 'mixed']),
# OrderedSet(['egalitarian', 'utilitarian']),
# OrderedSet(['egalitarian']),
# OrderedSet(['mixed', 'utilitarian']),
# OrderedSet(['mixed']),
# OrderedSet(['utilitarian']))"
EMPIRICAL_META_PAYOFFS = np.array(
    [
        [
            [2.764299878107763, 1.3754318747969299],
            [2.2615328446673537, 1.5902076414957933],
            [2.8053149442189285, 1.3916713428125362],
            [1.840795943449276, 1.83746234422587],
            [3.2260518454370057, 1.1444166400824596],
            [2.682269745885431, 1.3429529387657166],
            [3.769833944988581, 0.9458803413992027],
        ],
        [
            [2.2615328446673537, 1.5902076414957933],
            [2.2615328446673537, 1.5902076414957933],
            [1.840795943449276, 1.83746234422587],
            [1.840795943449276, 1.83746234422587],
            [2.682269745885431, 1.3429529387657166],
            [2.682269745885431, 1.3429529387657166],
            [0.7788309720751803, 0.4545001718301003],
        ],
        [
            [2.8053149442189285, 1.3916713428125362],
            [1.840795943449276, 1.83746234422587],
            [2.8053149442189285, 1.3916713428125362],
            [1.840795943449276, 1.83746234422587],
            [3.769833944988581, 0.9458803413992027],
            [0.7788309720751803, 0.4545001718301003],
            [3.769833944988581, 0.9458803413992027],
        ],
        [
            [1.840795943449276, 1.83746234422587],
            [1.840795943449276, 1.83746234422587],
            [1.840795943449276, 1.83746234422587],
            [1.840795943449276, 1.83746234422587],
            [0.7788309720751803, 0.4545001718301003],
            [0.7788309720751803, 0.4545001718301003],
            [0.7788309720751803, 0.4545001718301003],
        ],
        [
            [3.2260518454370057, 1.1444166400824596],
            [2.682269745885431, 1.3429529387657166],
            [3.769833944988581, 0.9458803413992027],
            [0.7788309720751803, 0.4545001718301003],
            [3.2260518454370057, 1.1444166400824596],
            [2.682269745885431, 1.3429529387657166],
            [3.769833944988581, 0.9458803413992027],
        ],
        [
            [2.682269745885431, 1.3429529387657166],
            [2.682269745885431, 1.3429529387657166],
            [0.7788309720751803, 0.4545001718301003],
            [0.7788309720751803, 0.4545001718301003],
            [2.682269745885431, 1.3429529387657166],
            [2.682269745885431, 1.3429529387657166],
            [0.7788309720751803, 0.4545001718301003],
        ],
        [
            [3.769833944988581, 0.9458803413992027],
            [0.7788309720751803, 0.4545001718301003],
            [3.769833944988581, 0.9458803413992027],
            [0.7788309720751803, 0.4545001718301003],
            [3.769833944988581, 0.9458803413992027],
            [0.7788309720751803, 0.4545001718301003],
            [3.769833944988581, 0.9458803413992027],
        ],
    ]
)
EMPIRICAL_META_PAYOFFS_P1 = EMPIRICAL_META_PAYOFFS[:, :, 0]
EMPIRICAL_META_PAYOFFS_P2 = EMPIRICAL_META_PAYOFFS[:, :, 1]


def sweep_and_plot_alpha(u1_, u2_, plot=False):
    nP = u1_.shape[0] * u1_.shape[1]
    alpha_list = np.logspace(-4, 2, 100)
    pi_list = np.zeros((0, nP))
    for alpha in alpha_list:
        try:
            _, _, pi, _, _ = alpharank.compute([u1_, u2_], alpha=alpha)
            pi_list = np.vstack((pi_list, pi))
            print("pi", np.argmax(pi))
        except ValueError:
            pass

    marginals = utils.get_alpharank_marginals([u1_, u2_], pi_list[-1])
    marginals = [marginals[0].round(2), marginals[1].round(2)]
    print("marginals", marginals)
    # pdb.set_trace()
    if plot:
        plt.plot(pi_list)
        plt.show()
    return pi_list[-1]


def sweep_and_plot_epsilon(u1_, u2_):
    nP = u1_.shape[0] * u1_.shape[1]
    eps_list = np.linspace(0, 1, 100)
    pi_list = np.zeros((0, nP))
    for eps in eps_list:
        try:
            _, _, pi, _, _ = alpharank.compute(
                [u1_, u2_], use_inf_alpha=True, inf_alpha_eps=eps
            )
            pi_list = np.vstack((pi_list, pi))
        except ValueError:
            pass
    plt.plot(pi_list)
    plt.show()


def exploitation_value(ui, u_mi, si, vi, normalize=True):
    counterpart_payoffs = np.dot(u_mi, si)
    exploiter = np.argmax(counterpart_payoffs)
    exploitation_value = np.dot(si, ui)[exploiter] - vi
    if normalize:
        exploitation_value /= ui.max() - ui.min()
    return exploitation_value


def is_dominated(u1, u2, v1, v2):
    better_than_v1 = u1 > v1
    better_than_v2 = u2 > v2
    where_dominated = better_than_v1 * better_than_v2
    is_dominated_anyhere = where_dominated.sum() > 0
    return is_dominated_anyhere


def evaluate_game_and_profile(u1, u2, s1, s2):
    v1_ = np.dot(s1, np.dot(u1, s2))
    v2_ = np.dot(s2, np.dot(u2, s1))
    exploitation_value_1 = exploitation_value(u1, u2, s1, v1_)
    exploitation_value_2 = exploitation_value(u2, u1, s2, v2_)
    is_dominated_ = is_dominated(u1, u2, v1_, v2_)
    return is_dominated_, exploitation_value_1, exploitation_value_2


def get_profile_from_meta_solver(u1, u2, meta_solver="alpharank"):
    if meta_solver == "alpharank":
        # joint_arank = sweep_and_plot_alpha(u1, u2)
        joint_arank = alpharank.sweep_pi_vs_alpha([u1, u2])
        s1, s2 = utils.get_alpharank_marginals([u1, u2], joint_arank)
    elif meta_solver == "rd":
        nA1, nA2 = u1.shape
        prd_initial_strategies = [
            np.random.dirichlet(alpha=np.ones(nA1) / nA1),
            np.random.dirichlet(alpha=np.ones(nA2) / nA2),
        ]
        s1, s2 = projected_replicator_dynamics.projected_replicator_dynamics(
            [u1, u2],
            prd_gamma=0.0,
            prd_initial_strategies=prd_initial_strategies,
        )
    return s1, s2


# def asymmetric_mcp_meta_solver_eval(
#     upper_bound=10,
#     grid_size=100,
#     meta_solver="alpharank",
#     game="bos",
#     plot=False,
# ):
#     k = 5
#     extra_points_labels = ["ks", "nash", "egal", "util", "meta"]
#     meta_distances = np.zeros((grid_size, 4))
#     payoff_profile = np.zeros((grid_size, 5, 2))
#     asymmetries = np.linspace(0, upper_bound, grid_size)
#     for rep, asymmetry in enumerate(asymmetries):
#         if game == "bos":
#             d1, d2 = 0.0, 0.0
#             base_payoff_1 = np.array(
#                 [[2 + asymmetry, 0.0], [0.0, 1.0 + expit(k * asymmetry) - 0.5]]
#             )
#             base_payoff_2 = np.array(
#                 [[1.0, 0.0], [0.0, 2 - expit(asymmetry * k) + 0.5]]
#             )
#         elif game == "chicken":
#             d1, d2 = -5, -5
#             base_payoff_1 = np.array(
#                 [
#                     [1.0, 0.0 + expit(asymmetry * k) - 0.5],
#                     [2 + asymmetry, -5.0],
#                 ]
#             )
#             base_payoff_2 = np.array(
#                 [[1.0, 2 - expit(asymmetry * k) + 0.5], [0.0, -5.0]]
#             )
#
#         s1, s2 = get_profile_from_meta_solver(
#             base_payoff_1, base_payoff_2, meta_solver=meta_solver
#         )
#         print(s1.round(2), s2.round(2))
#         u1_meta = np.dot(s1, np.dot(base_payoff_1, s2))
#         u2_meta = np.dot(s2, np.dot(base_payoff_2.T, s1))
#
#         (
#             (u1_ks, u2_ks),
#             (u1_nash, u2_nash),
#             (u1_egal, u2_egal),
#             (u1_util, u2_util),
#         ) = game_utils.optimize_welfare_discrete(
#             base_payoff_1, base_payoff_2, d1, d2, restrict_to_equilibria=True
#         )
#         extra_points = np.array(
#             [
#                 [u1_ks, u1_nash, u1_egal, u1_util, u1_meta],
#                 [u2_ks, u2_nash, u2_egal, u2_util, u2_meta],
#             ]
#         )
#         for i in range(5):
#             # meta_distances[rep, i] = np.mean((extra_points[:, i] - extra_points[:, 3])**2)
#             # meta_distances[rep, i] = np.allclose(extra_points[:, i], extra_points[:, 4], atol=0.2)
#             payoff_profile[rep, i, :] = extra_points[:, i] + np.random.normal(
#                 scale=0.05, size=2
#             )
#         print(f"welfare opt payoffs:\n{extra_points.round(2)}")
#         # if plot:
#         # game_utils.create_figure(base_payoff_1, base_payoff_2, None, show_points=True, extra_points=extra_points,
#         #                          extra_points_labels=extra_points_labels, fill=False)
#     if plot:
#         # plt.plot(meta_distances)
#         for i in range(payoff_profile.shape[1]):
#             # plt.scatter(payoff_profile[:, i, 0], payoff_profile[:, i, 1], label=extra_points_labels[i])
#             plt.plot(
#                 asymmetries,
#                 payoff_profile[:, i, 0],
#                 label=extra_points_labels[i],
#             )
#         plt.xlabel("xi")
#         plt.ylabel("player 1 payoff")
#         plt.legend()
#         plt.show()
#     return


# def random_mcp_meta_solver_eval(
#     n_rep=1, meta_solver="alpharank", game="bos", plot=False
# ):
#     restrict_to_equilibria = True
#     if game == "bos":
#         d1, d2 = 0.0, 0.0
#         base_payoff_1 = np.array([[2.0, 0.0], [0.0, 1.0]])
#         base_payoff_2 = np.array([[1.0, 0.0], [0.0, 2.0]])
#     elif game == "chicken":
#         d1, d2 = -5.0, -5.0
#         base_payoff_1 = np.array([[1.0, 0.0], [2.0, -5.0]])
#         base_payoff_2 = np.array([[1.0, 2.0], [0.0, -5.0]])
#     elif game == "bospd":
#         d1, d2 = 0.0, 0.0
#         base_payoff_1 = np.array(
#             [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
#         )
#         base_payoff_2 = np.array(
#             [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
#         )
#     elif game == "random":
#         nA = 5
#         base_payoff_1 = np.zeros((nA, nA))
#         base_payoff_2 = np.zeros((nA, nA))
#         d1, d2 = None, None
#         restrict_to_equilibria = False
#
#     extra_points_labels = ["ks", "nash", "egal", "util", "meta"]
#     meta_distances = np.zeros((0, len(extra_points_labels) - 1))
#     for rep in range(n_rep):
#         perturbation_1 = np.random.normal(scale=0.5, size=2)
#         perturbation_2 = np.random.normal(scale=0.5, size=2)
#
#         perturbed_payoff_1 = copy.copy(base_payoff_1)
#         perturbed_payoff_2 = copy.copy(base_payoff_2)
#
#         if game in ["bos", "bospd"]:
#             perturbed_payoff_1[[0, 1], [0, 1]] += perturbation_1
#             perturbed_payoff_2[[0, 1], [0, 1]] += perturbation_2
#             if game == "bospd":
#                 best_bos = np.max(
#                     (
#                         perturbed_payoff_1[:2, :2].max(),
#                         perturbed_payoff_2[:2, :2].max(),
#                     )
#                 )
#                 worst_bos = np.min(
#                     (
#                         perturbed_payoff_1[:2, :2].min(),
#                         perturbed_payoff_2[:2, :2].min(),
#                     )
#                 )
#                 mutual_defect_payoff = worst_bos - 1
#                 # defection_payoff = best_bos + 1.
#                 # sucker_payoff = worst_bos - 2.
#                 # ToDo: these are assuming iterated bospd, where defections are grim triggered
#                 defection_payoff = mutual_defect_payoff
#                 sucker_payoff = mutual_defect_payoff
#                 perturbed_payoff_1[2, 2] = mutual_defect_payoff
#                 perturbed_payoff_2[2, 2] = mutual_defect_payoff
#                 perturbed_payoff_1[2, :2] = defection_payoff
#                 perturbed_payoff_2[:2, 2] = defection_payoff
#                 perturbed_payoff_2[2, :2] = sucker_payoff
#                 perturbed_payoff_1[:2, 2] = sucker_payoff
#
#         elif game == "chicken":
#             perturbed_payoff_1[[0, 1], [1, 0]] += perturbation_1
#             perturbed_payoff_2[[0, 1], [1, 0]] += perturbation_2
#         elif game == "random":
#             perturbed_payoff_1 = base_payoff_1 + np.random.normal(
#                 size=(nA, nA)
#             )
#             perturbed_payoff_2 = base_payoff_2 + np.random.normal(
#                 size=(nA, nA)
#             )
#             multiple_equilibria = (
#                 len(
#                     list(
#                         nash.Game(
#                             perturbed_payoff_1, perturbed_payoff_2
#                         ).support_enumeration()
#                     )
#                 )
#                 > 1
#             )
#             # multiple_equilibria = True
#
#         s1, s2 = get_profile_from_meta_solver(
#             perturbed_payoff_1, perturbed_payoff_2, meta_solver=meta_solver
#         )
#         print(s1.round(2), s2.round(2))
#         u1_meta = np.dot(s1, np.dot(perturbed_payoff_1, s2))
#         u2_meta = np.dot(s2, np.dot(perturbed_payoff_2.T, s1))
#
#         if game != "random" or (game == "random" and multiple_equilibria):
#             (
#                 (u1_ks, u2_ks),
#                 (u1_nash, u2_nash),
#                 (u1_egal, u2_egal),
#                 (u1_util, u2_util),
#             ) = game_utils.optimize_welfare_discrete(
#                 perturbed_payoff_1,
#                 perturbed_payoff_2,
#                 d1,
#                 d2,
#                 restrict_to_equilibria=restrict_to_equilibria,
#             )
#             extra_points = np.array(
#                 [
#                     [u1_ks, u1_nash, u1_egal, u1_util, u1_meta],
#                     [u2_ks, u2_nash, u2_egal, u2_util, u2_meta],
#                 ]
#             )
#             n_welfare = len(extra_points[0]) - 1
#             meta_distances_rep = np.zeros(n_welfare)
#             for i in range(n_welfare):
#                 # meta_distances[rep, i] = np.mean((extra_points[:, i] - extra_points[:, 3])**2)
#                 meta_distances_rep[i] = np.allclose(
#                     extra_points[:, i], extra_points[:, n_welfare], atol=0.2
#                 )
#             meta_distances = np.vstack((meta_distances, meta_distances_rep))
#             print(f"welfare opt payoffs:\n{extra_points.round(2)}")
#             if plot:
#                 game_utils.create_figure(
#                     perturbed_payoff_1,
#                     perturbed_payoff_2,
#                     None,
#                     show_points=True,
#                     extra_points=extra_points,
#                     extra_points_labels=extra_points_labels,
#                     fill=False,
#                 )
#     print(meta_distances.mean(axis=0).round(2))
#     return


def random_meta_solver_eval(nA=3, n_rep=20, meta_solver="alpharank"):
    is_dominated_list = np.zeros(n_rep)
    exploitation_1_list = np.zeros(n_rep)
    exploitation_2_list = np.zeros(n_rep)

    for rep in range(n_rep):
        print(rep)
        u1_ = np.random.random(size=(nA, nA))
        u2_ = np.random.random(size=(nA, nA))

        s1, s2 = get_profile_from_meta_solver(
            u1_, u2_, meta_solver=meta_solver
        )

        (
            is_dominated_,
            exploitation_value_1,
            exploitation_value_2,
        ) = evaluate_game_and_profile(u1_, u2_, s1, s2)
        is_dominated_list[rep] = is_dominated_
        exploitation_1_list[rep] = exploitation_value_1
        exploitation_2_list[rep] = exploitation_value_2

    pct_dominated = is_dominated_list.mean()
    exploitation_1_mean = exploitation_1_list.mean()
    exploitation_2_mean = exploitation_2_list.mean()

    print(pct_dominated, exploitation_1_mean, exploitation_2_mean)

    return


if __name__ == "__main__":
    # ToDo: do the same thing but for continuous barg meta games
    # ToDo: incorporate utilitarian outcome
    # sweep_and_plot_alpha(SMALL_META_PAYOUT_1, SMALL_META_PAYOUT_1, plot=True)
    sweep_and_plot_alpha(
        EMPIRICAL_META_PAYOFFS_P1, EMPIRICAL_META_PAYOFFS_P2, plot=True
    )
    # random_mcp_meta_solver_eval(n_rep=20, game='bos', meta_solver='rd')
    # asymmetric_mcp_meta_solver_eval(game='bos', grid_size=10, plot=True, meta_solver='rd')
    # asymmetric_mcp_meta_solver_eval(game='chicken', plot=True)
