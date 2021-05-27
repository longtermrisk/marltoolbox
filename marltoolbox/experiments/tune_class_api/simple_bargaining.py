import numpy as np

# import nashpy as nash
import torch
import matplotlib.pyplot as plt
from functools import partial

# from prd import prd
from collections import Counter
import pdb
import argparse
import copy

# from feasible_set_figure import optimize_welfare_discrete
# from meta_solver_exploration import sweep_and_plot_alpha
from open_spiel.python.algorithms import projected_replicator_dynamics
from open_spiel.python.egt import alpharank
from open_spiel.python.algorithms.psro_v2 import utils
import matplotlib.pyplot as plt
from marltoolbox.utils import miscellaneous

np.set_printoptions(suppress=True)

IPD_PAYOUT = torch.Tensor([[2, -3], [3, -1]])
ASYM_IPD_PAYOUT_1 = torch.Tensor([[2, -3], [20, -1]])
ASYM_IPD_PAYOUT_2 = torch.Tensor([[2, 3], [-2.5, -1]])

DISAGREEMENT = 0.0
MISCOORDINATION = 0.8
# CHICKEN_PAYOUT_1 = np.array([[1., -2.], [2, -10]])
# CHICKEN_PAYOUT_2 = np.array([[1., 2.], [-2, -10]])

BOTS_PAYOUT_1 = torch.Tensor([[4.0, MISCOORDINATION], [MISCOORDINATION, 2.0]])
BOTS_PAYOUT_2 = torch.Tensor([[1.5, MISCOORDINATION], [MISCOORDINATION, 2.0]])
META_PAYOUT_1 = torch.Tensor(
    [
        [
            4.0,
            MISCOORDINATION,
            MISCOORDINATION,
            4.0,
            4.0,
            MISCOORDINATION,
            4.0,
        ],
        [
            MISCOORDINATION,
            2.0,
            MISCOORDINATION,
            2.0,
            MISCOORDINATION,
            2.0,
            2.0,
        ],
        [
            MISCOORDINATION,
            MISCOORDINATION,
            3.0,
            MISCOORDINATION,
            3.0,
            3.0,
            3.0,
        ],
        [4.0, 2.0, MISCOORDINATION, 3.0, 4.0, 2.0, 3.0],
        [4.0, MISCOORDINATION, 3.0, 4.0, 3.5, 3.0, 3.5],
        [MISCOORDINATION, 2.0, 3.0, 2.0, 3.0, 2.5, 2.5],
        [4.0, 2.0, 3.0, 3.0, 3.5, 2.5, 3.0],
    ]
)
META_PAYOUT_2 = torch.Tensor(
    [
        [1.0, MISCOORDINATION, MISCOORDINATION, 1.0, 1, MISCOORDINATION, 1.0],
        [
            MISCOORDINATION,
            2.0,
            MISCOORDINATION,
            2.0,
            MISCOORDINATION,
            2.0,
            2.0,
        ],
        [
            MISCOORDINATION,
            MISCOORDINATION,
            1.5,
            MISCOORDINATION,
            1.5,
            1.5,
            1.5,
        ],
        [1.0, 2.0, MISCOORDINATION, 1.5, 1.0, 2.0, 1.5],
        [1.0, MISCOORDINATION, 1.5, 1.0, 1.25, 1.5, 1.75],
        [MISCOORDINATION, 2.0, 1.5, 2.0, 1.5, 1.75, 1.75],
        [1.0, 2.0, 1.5, 1.5, 1.25, 1.75, 1.5],
    ]
)

# SMALL_META_PAYOUT_1 = np.array([[4., 0., 4.], [0., 2., 2.], [4., 2., 3.]])
# SMALL_META_PAYOUT_2 = np.array([[1., 0., 1.], [0., 2., 2.], [1., 2., 1.5]])
MULTIPLIER = 0.2
P11, P12, P21, P22 = np.array([3, 9, 7, 2]) * MULTIPLIER
TEMPERATURE = 20
G = 3


def bots(gamma=0.96):
    dims = [5, 5]
    payout_mat_1 = BOTS_PAYOUT_1
    payout_mat_2 = BOTS_PAYOUT_2

    def Ls(th):
        p_1_0 = torch.sigmoid(th[0][0:1])
        p_2_0 = torch.sigmoid(th[1][0:1])
        p = torch.cat(
            [
                p_1_0 * p_2_0,
                p_1_0 * (1 - p_2_0),
                (1 - p_1_0) * p_2_0,
                (1 - p_1_0) * (1 - p_2_0),
            ]
        )
        p_1 = torch.reshape(torch.sigmoid(th[0][1:5]), (4, 1))
        p_2 = torch.reshape(torch.sigmoid(th[1][1:5]), (4, 1))
        P = torch.cat(
            [
                p_1 * p_2,
                p_1 * (1 - p_2),
                (1 - p_1) * p_2,
                (1 - p_1) * (1 - p_2),
            ],
            dim=1,
        )
        M = -torch.matmul(p, torch.inverse(torch.eye(4) - gamma * P))
        L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
        return [L_1, L_2]

    return dims, Ls


def ipd(gamma=0.96, asymmetric=False):
    dims = [5, 5]
    if asymmetric:
        payout_mat_1 = ASYM_IPD_PAYOUT_1
        payout_mat_2 = ASYM_IPD_PAYOUT_2
    else:
        payout_mat_1 = IPD_PAYOUT
        payout_mat_2 = payout_mat_1.T

    def Ls(th):
        p_1_0 = torch.sigmoid(th[0][0:1])
        p_2_0 = torch.sigmoid(th[1][0:1])
        p = torch.cat(
            [
                p_1_0 * p_2_0,
                p_1_0 * (1 - p_2_0),
                (1 - p_1_0) * p_2_0,
                (1 - p_1_0) * (1 - p_2_0),
            ]
        )
        p_1 = torch.reshape(torch.sigmoid(th[0][1:5]), (4, 1))
        p_2 = torch.reshape(torch.sigmoid(th[1][1:5]), (4, 1))
        P = torch.cat(
            [
                p_1 * p_2,
                p_1 * (1 - p_2),
                (1 - p_1) * p_2,
                (1 - p_1) * (1 - p_2),
            ],
            dim=1,
        )
        M = -torch.matmul(p, torch.inverse(torch.eye(4) - gamma * P))
        L_1 = torch.matmul(M, torch.reshape(payout_mat_1, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(payout_mat_2, (4, 1)))
        return [L_1, L_2]

    return dims, Ls


def meta():
    dims = [7, 7]
    payout_mat_1 = META_PAYOUT_1
    payout_mat_2 = META_PAYOUT_2

    fair1 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 6
    fair2 = torch.Tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 6

    def Ls(th):
        p1 = torch.nn.functional.softmax(th[0])
        p2 = torch.nn.functional.softmax(th[1])
        L_1 = -torch.dot(p2, torch.matmul(payout_mat_1, p1))
        L_2 = -torch.dot(p1, torch.matmul(payout_mat_2, p2))
        return [L_1, L_2]

    def Ls2(th):
        p1 = torch.nn.functional.softmax(th[0])
        p2 = torch.nn.functional.softmax(th[1])
        L_1 = -torch.dot(fair1, torch.matmul(payout_mat_1, p1))
        L_2 = -torch.dot(fair2, torch.matmul(payout_mat_2, p2))
        return [L_1, L_2]

    return dims, Ls, Ls2


# def tandem():
#   dims = [1, 1]
#   def Ls(th):
#     x, y = th
#     L_1 = (x+y)**2-2*x
#     L_2 = (x+y)**2-2*y
#     return [L_1, L_2]
#   return dims, Ls


# def get_policies_as_matrix(th, nA):
#   # Parameters for initial actions
#   p_1_0 = torch.nn.functional.softmax(th[0][0:nA])
#   p_2_0 = torch.nn.functional.softmax(th[1][0:nA])
#   p = torch.flatten(torch.ger(p_1_0, p_2_0))
#
#   # Parameters for actions conditional on previous action profile
#   p_1_lst = []
#   p_2_lst = []
#   joint_p_lst = []
#   for i in range(1, nA**2+1):  # Loop through action profile indices
#     p_1_i = torch.nn.functional.softmax(th[0][(nA*i):(nA*i+nA)])
#     p_2_i = torch.nn.functional.softmax(th[1][(nA*i):(nA*i+nA)])
#     p_1_lst.append(p_1_i)
#     p_2_lst.append(p_2_i)
#
#     # joint_p gives probability of each action profile conditional on previous profile (torch.ger is outer product)
#     joint_p = torch.ger(p_1_i, p_2_i).reshape(nA**2, 1)
#     joint_p_lst.append(joint_p.T)
#
#   p_1 = torch.cat(p_1_lst)
#   p_2 = torch.cat(p_2_lst)
#   P = torch.cat(joint_p_lst, dim=0)
#   return p, p_1, p_2, P


# def bipd_loss(th, u1, u2, gamma):
#     """
#     Get closed form value functions for each player given policy profile parameter
#     (negated since these algorithms act on losses).
#     V_i = u_i + \gamma * PV_i
#       => V_i = (I - \gamma P)^-1 u_i;
#     L_i = -p.V_i, where p is distribution of initial action profiles
#     :param th: Tensor of parameters for each policy
#     :param u1: Player 1 payoff matrix
#     :param u2: Player 2 payoff matrix
#     """
#     p, p_1, p_2, P = get_bipd_policies_as_matrix(th)
#     M = -torch.matmul(p, torch.inverse(torch.eye(9) - gamma * P))
#     L_1 = torch.matmul(M, torch.reshape(u1, (9, 1)))
#     L_2 = torch.matmul(M, torch.reshape(u2, (9, 1)))
#     return [L_1, L_2]
#
#
# def bipd(payout_mat_1, payout_mat_2, gamma=0.96):
#     dims = [30, 30]
#     # ToDo: fiddling with payoffs, may not be original bipd payoffs
#
#     Ls = partial(bipd_loss, u1=payout_mat_1, u2=payout_mat_2, gamma=gamma)
#
#     return dims, Ls


def continuous_bargaining(temperature=None):
    """
    Utility functions of the form
    1[ \theta_1 > \tau_2 ] * 1[ \theta_2 < \tau_1] * [ theta_i^P_i1 + (1 - theta_i)*P_i2 ]
    Policy params are allocations \theta_i and acceptance thresholds \theta_i.
    """

    dims = [2, 2]
    if temperature is None:
        temperature = TEMPERATURE

    def Ls(th_, soft_cutoff=True):
        allocation_1 = torch.sigmoid(th_[0][0])
        allocation_2 = torch.sigmoid(th_[1][0])
        cutoff_1 = torch.sigmoid(th_[0][1])
        cutoff_2 = torch.sigmoid(th_[1][1])

        soft_indicator_1 = torch.sigmoid(
            temperature * (allocation_1 - cutoff_2)
        )
        soft_indicator_2 = torch.sigmoid(
            temperature * (cutoff_1 - allocation_2)
        )
        soft_indicator = soft_indicator_1 * soft_indicator_2

        if not soft_cutoff:
            if (allocation_1 - cutoff_2) < 0 or (cutoff_1 - allocation_2) < 0:
                return [torch.tensor(0.0), torch.tensor(0.0)]
            else:
                soft_indicator = 1.0

        l11 = torch.log(torch.pow(allocation_1 + allocation_2, P11) + 1)
        l12 = torch.log(
            torch.pow(1 - allocation_1 + G * (1 - allocation_2), P12) + 1
        )
        l21 = torch.log(torch.pow(G * allocation_1 + allocation_2, P21) + 1)
        l22 = torch.log(torch.pow(2 - (allocation_1 + allocation_2), P22) + 1)

        agreement_payoff_1 = l11 + l12
        agreement_payoff_2 = l21 + l22

        L1 = -(
            soft_indicator * agreement_payoff_1
            + (1 - soft_indicator) * DISAGREEMENT
        )
        L2 = -(
            soft_indicator * agreement_payoff_2
            + (1 - soft_indicator) * DISAGREEMENT
        )

        return [L1, L2]

    return dims, Ls


# @markdown Gradient computations for each algorithm.
def init_th(dims, std):
    th = []
    for i in range(len(dims)):
        if std > 0:
            init = torch.nn.init.normal_(
                torch.empty(dims[i], requires_grad=True), std=std
            )
        else:
            init = torch.zeros(dims[i], requires_grad=True)
        th.append(init)
    return th


def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True)[0]
    return grad


def get_hessian(th, grad_L, diag=True, off_diag=True):
    n = len(th)
    H = []
    for i in range(n):
        row_block = []
        for j in range(n):
            if (i == j and diag) or (i != j and off_diag):
                block = [
                    torch.unsqueeze(
                        get_gradient(grad_L[i][i][k], th[j]), dim=0
                    )
                    for k in range(len(th[i]))
                ]
                row_block.append(torch.cat(block, dim=0))
            else:
                row_block.append(torch.zeros(len(th[i]), len(th[j])))
        H.append(torch.cat(row_block, dim=1))
    return torch.cat(H, dim=0)


def update_th(
    th,
    Ls,
    alpha,
    algo,
    Ls2=None,
    a=0.5,
    b=0.1,
    gam=1,
    ep=0.1,
    lss_lam=0.1,
    tradeoff=None,
):
    n = len(th)
    losses = Ls(th)

    # Compute gradients
    grad_L = [
        [get_gradient(losses[j], th[i]) for j in range(n)] for i in range(n)
    ]
    if Ls2 is not None:
        losses2 = Ls2(th)
        grad_L2 = [get_gradient(losses2[i], th[i]) for i in range(n)]
    if algo == "la":
        terms = [
            sum(
                [
                    torch.dot(grad_L[j][i], grad_L[j][j].detach())
                    for j in range(n)
                    if j != i
                ]
            )
            for i in range(n)
        ]
        grads = [
            grad_L[i][i] - alpha * get_gradient(terms[i], th[i])
            for i in range(n)
        ]
    elif algo == "lola":
        terms = [
            sum(
                [
                    torch.dot(grad_L[j][i], grad_L[j][j])
                    for j in range(n)
                    if j != i
                ]
            )
            for i in range(n)
        ]
        grads = [
            grad_L[i][i] - alpha * get_gradient(terms[i], th[i])
            for i in range(n)
        ]
    elif algo == "sos":
        terms = [
            sum(
                [
                    torch.dot(grad_L[j][i], grad_L[j][j].detach())
                    for j in range(n)
                    if j != i
                ]
            )
            for i in range(n)
        ]
        xi_0 = [
            grad_L[i][i] - alpha * get_gradient(terms[i], th[i])
            for i in range(n)
        ]
        chi = [
            get_gradient(
                sum(
                    [
                        torch.dot(grad_L[j][i].detach(), grad_L[j][j])
                        for j in range(n)
                        if j != i
                    ]
                ),
                th[i],
            )
            for i in range(n)
        ]
        # Compute p
        dot = torch.dot(-alpha * torch.cat(chi), torch.cat(xi_0))
        p1 = (
            1
            if dot >= 0
            else min(1, -a * torch.norm(torch.cat(xi_0)) ** 2 / dot)
        )
        xi = torch.cat([grad_L[i][i] for i in range(n)])
        xi_norm = torch.norm(xi)
        p2 = xi_norm ** 2 if xi_norm < b else 1
        p = min(p1, p2)
        grads = [xi_0[i] - p * alpha * chi[i] for i in range(n)]
    elif algo == "sos_tradeoff":
        terms = [
            sum(
                [
                    torch.dot(grad_L[j][i], grad_L[j][j].detach())
                    for j in range(n)
                    if j != i
                ]
            )
            for i in range(n)
        ]
        xi_0 = [
            grad_L[i][i] - alpha * get_gradient(terms[i], th[i])
            for i in range(n)
        ]
        chi = [
            get_gradient(
                sum(
                    [
                        torch.dot(grad_L[j][i].detach(), grad_L[j][j])
                        for j in range(n)
                        if j != i
                    ]
                ),
                th[i],
            )
            for i in range(n)
        ]
        # Compute p
        dot = torch.dot(-alpha * torch.cat(chi), torch.cat(xi_0))
        p1 = (
            1
            if dot >= 0
            else min(1, -a * torch.norm(torch.cat(xi_0)) ** 2 / dot)
        )
        xi = torch.cat([grad_L[i][i] for i in range(n)])
        xi_norm = torch.norm(xi)
        p2 = xi_norm ** 2 if xi_norm < b else 1
        p = min(p1, p2)
        grads = [
            tradeoff * (xi_0[i] - p * alpha * chi[i])
            + (1 - tradeoff) * grad_L2[i]
            for i in range(n)
        ]
    elif algo == "sga":
        xi = torch.cat([grad_L[i][i] for i in range(n)])
        ham = torch.dot(xi, xi.detach())
        H_t_xi = [get_gradient(ham, th[i]) for i in range(n)]
        H_xi = [
            get_gradient(
                sum(
                    [
                        torch.dot(grad_L[j][i], grad_L[j][j].detach())
                        for j in range(n)
                    ]
                ),
                th[i],
            )
            for i in range(n)
        ]
        A_t_xi = [H_t_xi[i] / 2 - H_xi[i] / 2 for i in range(n)]
        # Compute lambda (sga with alignment)
        dot_xi = torch.dot(xi, torch.cat(H_t_xi))
        dot_A = torch.dot(torch.cat(A_t_xi), torch.cat(H_t_xi))
        d = sum([len(th[i]) for i in range(n)])
        lam = torch.sign(dot_xi * dot_A / d + ep)
        grads = [grad_L[i][i] + lam * A_t_xi[i] for i in range(n)]
    elif algo == "co":
        xi = torch.cat([grad_L[i][i] for i in range(n)])
        ham = torch.dot(xi, xi.detach())
        grads = [
            grad_L[i][i] + gam * get_gradient(ham, th[i]) for i in range(n)
        ]
    elif algo == "eg":
        th_eg = [
            th[i] - alpha * get_gradient(losses[i], th[i]) for i in range(n)
        ]
        losses_eg = Ls(th_eg)
        grads = [get_gradient(losses_eg[i], th_eg[i]) for i in range(n)]
    elif algo == "cgd":  # Slow implementation (matrix inversion)
        dims = [len(th[i]) for i in range(n)]
        xi = torch.cat([grad_L[i][i] for i in range(n)])
        H_o = get_hessian(th, grad_L, diag=False)
        grad = torch.matmul(
            torch.inverse(torch.eye(sum(dims)) + alpha * H_o), xi
        )
        grads = [grad[sum(dims[:i]) : sum(dims[: i + 1])] for i in range(n)]
    elif algo == "lss":  # Slow implementation (matrix inversion)
        dims = [len(th[i]) for i in range(n)]
        xi = torch.cat([grad_L[i][i] for i in range(n)])
        H = get_hessian(th, grad_L)
        if torch.det(H) == 0:
            inv = torch.inverse(
                torch.matmul(H.T, H) + lss_lam * torch.eye(sum(dims))
            )
            H_inv = torch.matmul(inv, H.T)
        else:
            H_inv = torch.inverse(H)
        grad = (
            torch.matmul(torch.eye(sum(dims)) + torch.matmul(H.T, H_inv), xi)
            / 2
        )
        grads = [grad[sum(dims[:i]) : sum(dims[: i + 1])] for i in range(n)]
    else:  # Naive Learning
        grads = [grad_L[i][i] for i in range(n)]

    # Update theta
    with torch.no_grad():
        for i in range(n):
            th[i] -= alpha * grads[i]
    return th, losses


def continuous_bargaining_meta_penalty(th_):
    penalty = 1 * torch.norm(th_[0][1:] - th_[1][1:])
    return penalty


def learn_bots_pd(
    num_runs=20,
    method="lola",
    game="bots_pd",
    save=False,
    tradeoff=None,
    U1=None,
    U2=None,
    std=1.0,
    min_joint_perf=-1e9,
    temperature=TEMPERATURE,
    jitter=0.0,
    num_epochs=200,
    alpha=1.0,
    verbose=False,
    lr_decay=False,
    hard_cutoff_in_eval=False,
):
    gamma = 0.96
    Ls2 = None

    if game == "pd":
        payout_mat_1 = IPD_PAYOUT
        payout_mat_2 = payout_mat_1.T
        dims, Ls = ipd(gamma)

    elif game == "bos":
        payout_mat_1 = BOTS_PAYOUT_1
        payout_mat_2 = BOTS_PAYOUT_2
        dims, Ls = bots(gamma)

    elif game == "meta":
        dims, Ls, Ls2 = meta()

    elif game == "continuous":

        dims, Ls = continuous_bargaining(temperature)

    # elif game == 'continuous_unexploitable':
    #     dims, Ls = continuous_bargaining_unexploitable()
    #
    # elif game == 'continuous_meta':
    #     dims, Ls = continuous_bargaining_meta()
    #
    # elif game == 'matrix':
    #     U1_tensor = torch.tensor(U1).float()
    #     U2_tensor = torch.tensor(U2).float()
    #     dims, Ls = matrix(U1_tensor, U2_tensor)
    #
    # elif game == 'aipd':
    #     dims, Ls = ipd(gamma, asymmetric=True)

    # alpha = 1
    # std = 1

    # List of policy profiles thetai generated by several independent runs by player i
    theta1_dbn = []
    theta2_dbn = []

    losses_out = np.zeros((num_runs, num_epochs))

    def _get_rewards(losses, th):
        if game in ["meta", "continuous", "continuous_unexploitable"]:
            r_pl1 = -losses[0].detach().numpy()
            r_pl2 = -losses[1].detach().numpy()
        elif game in ["continuous_meta"]:
            pen = continuous_bargaining_meta_penalty(th).detach().numpy()
            losses_ = Ls(th)
            r_pl1 = -losses_[0].detach().numpy() + pen
            r_pl2 = -losses_[1].detach().numpy() + pen
        else:
            r_pl1 = -losses[0].detach().numpy() * (1 - gamma)
            r_pl2 = -losses[1].detach().numpy() * (1 - gamma)
        return r_pl1, r_pl2

    # Lists of (negative) losses for self play
    final_losses_1 = []
    final_losses_2 = []

    # List of (negative) losses for cross play
    cross_play_losses_1 = []
    cross_play_losses_2 = []
    th_prev = None
    for i in range(num_runs):
        print("run n°", i)
        r_pl1, r_pl2 = min_joint_perf, min_joint_perf
        trial = 0
        while r_pl1 <= min_joint_perf and r_pl2 <= min_joint_perf:
            trial += 1
            print("trial", trial, "rewards", r_pl1, r_pl2)

            seed = miscellaneous.get_random_seeds(1)[0]
            torch.manual_seed(seed)
            np.random.seed(seed)

            th = init_th(dims, std)
            for k in range(num_epochs):
                if lr_decay:
                    th, losses = update_th(
                        th,
                        Ls,
                        alpha * (num_epochs - k) / num_epochs,
                        method,
                        Ls2=Ls2,
                        tradeoff=tradeoff,
                    )
                else:
                    th, losses = update_th(
                        th, Ls, alpha, method, Ls2=Ls2, tradeoff=tradeoff
                    )
                # Negating so that higher => better!
                losses_out[i, k] = -(1 - gamma) * losses[0].data.numpy()

            self_losses = Ls(th, soft_cutoff=hard_cutoff_in_eval)
            r_pl1, r_pl2 = _get_rewards(self_losses, th)

        final_losses_1.append(r_pl1)
        final_losses_2.append(r_pl2)
        theta1_dbn.append(th[0])
        theta2_dbn.append(th[1])
        print("th", th)

        # Evaluate player 1 policy learned in previous run against player 2 policy learned in this run
        if i > 0:
            th_cross = [th_prev[0], th[1]]
            cross_losses = Ls(th_cross, soft_cutoff=hard_cutoff_in_eval)
            r_pl1_cross, r_pl2_cross = _get_rewards(cross_losses, th_cross)
            cross_play_losses_1.append(r_pl1_cross)
            cross_play_losses_2.append(r_pl2_cross)

            th_cross = [th[0], th_prev[1]]
            cross_losses = Ls(th_cross, soft_cutoff=hard_cutoff_in_eval)
            r_pl1_cross, r_pl2_cross = _get_rewards(cross_losses, th_cross)
            cross_play_losses_1.append(r_pl1_cross)
            cross_play_losses_2.append(r_pl2_cross)

        th_prev = th

        if verbose:
            print("losses_out[i, :]", losses_out[i, :])

    def _format_values(values):
        if game not in [
            "meta",
            "continuous",
            "continuous_unexploitable",
            "continuous_meta",
        ]:
            values = [v[0] for v in values]
        values = np.array(values)
        return values

    final_losses_1 = _format_values(final_losses_1)
    final_losses_2 = _format_values(final_losses_2)
    cross_play_losses_1 = _format_values(cross_play_losses_1)
    cross_play_losses_2 = _format_values(cross_play_losses_2)

    print(
        "self-play r_pl_1 mean std",
        final_losses_1.mean(),
        "std",
        final_losses_1.std(),
    )
    print(
        "self-play r_pl_2 mean std",
        final_losses_2.mean(),
        "std",
        final_losses_2.std(),
    )
    print(
        "cross-play r_pl_1 mean std",
        cross_play_losses_1.mean(),
        "std",
        cross_play_losses_1.std(),
    )
    print(
        "cross-play r_pl_2 mean",
        cross_play_losses_2.mean(),
        "std",
        cross_play_losses_2.std(),
    )
    if save:
        np.save("cross_play_losses_1.npy", np.array(cross_play_losses_1))
        np.save("cross_play_losses_2.npy", np.array(cross_play_losses_2))
        torch.save(torch.stack(theta1_dbn), "theta1_dbn.pt")
        torch.save(torch.stack(theta2_dbn), "theta2_dbn.pt")

    def _add_jitter(values):
        values += np.random.normal(0.0, jitter, values.shape)
        return values

    final_losses_1 = _add_jitter(final_losses_1)
    final_losses_2 = _add_jitter(final_losses_2)
    cross_play_losses_1 = _add_jitter(cross_play_losses_1)
    cross_play_losses_2 = _add_jitter(cross_play_losses_2)

    # plt.scatter(x=cross_play_losses_1, y=cross_play_losses_2)
    # plt.scatter(x=final_losses_1, y=final_losses_2)
    plt.plot(
        cross_play_losses_1,
        cross_play_losses_2,
        markerfacecolor="none",
        markeredgecolor="#1f77b4",
        marker="o",
        linestyle="None",
    )
    plt.plot(
        final_losses_1,
        final_losses_2,
        markerfacecolor="none",
        markeredgecolor="#ff7f0e",
        marker="o",
        linestyle="None",
    )
    plt.legend(["cross-play", "self-play"])
    title = (
        f"env({game}) "
        f"algo({method}) "
        f"hard_cutoff({hard_cutoff_in_eval}) "
        f"fail({min_joint_perf}) "
        f"T({temperature}) "
        f"num_runs({num_runs})"
    )
    plt.suptitle(title)
    plt.xlim((-0.1, 3.0))
    plt.ylim((-0.1, 3.0))
    title = title.replace(" ", "_")
    plt.savefig(title + ".png")
    plt.show()

    return (
        final_losses_1,
        final_losses_2,
        cross_play_losses_1,
        cross_play_losses_2,
        theta1_dbn,
        theta2_dbn,
    )


def main(debug):

    scaling_training = 1
    learn_bots_pd(
        num_runs=40,
        #
        # method="ni",
        method="lola",
        # method="sos",
        #
        # game="pd",
        #
        # game="bos",
        # std=3.0,
        #
        game="continuous",
        # min_joint_perf=0.2,
        # min_joint_perf=2.0,
        min_joint_perf=2.25,
        #
        # temperature=1,
        jitter=0.02,
        #
        save=False,
        tradeoff=None,
        U1=None,
        U2=None,
        #
        # num_epochs=200 * scaling_training,
        alpha=1.0 / scaling_training,
        # verbose=True,
        num_epochs=200 * scaling_training,
        # 0.0581 -> 0.0597
        # lr_decay=True, # not helping
        hard_cutoff_in_eval=True,
    )


if __name__ == "__main__":
    debug_mode = True
    main(debug_mode)
