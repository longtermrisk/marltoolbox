######
# Code modified from:
# https://github.com/julianstastny/openspiel-social-dilemmas
######
import os
import random

import torch
import numpy as np
from ray import tune

from marltoolbox.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
    IteratedAsymBoS,
)


class SOSTrainer(tune.Trainable):
    def setup(self, config: dict):

        self.config = config
        self.gamma = self.config.get("gamma")
        self.learning_rate = self.config.get("lr")
        self.method = self.config.get("method")

        self._set_environment(self.config.get("env_name"))
        self.init_weigths(std=self.config.get("inital_weights_std"))

    def _set_environment(self, env_name):

        if env_name == "IteratedPrisonersDilemma":
            env_class = IteratedPrisonersDilemma
        elif env_name == "IteratedAsymBoS":
            env_class = IteratedAsymBoS
        else:
            raise NotImplementedError()

        payoff_matrix = env_class.PAYOFF_MATRIX
        self.dims = [5, 5]
        self.players_ids = env_class({}).players_ids
        self.exact_loss_function = (
            self._exact_loss_matrix_game_two_by_two_actions
        )

        self.payoff_matrix_player_row = torch.tensor(
            payoff_matrix[:, :, 0]
        ).float()
        self.payoff_matrix_player_col = torch.tensor(
            payoff_matrix[:, :, 1]
        ).float()

    def step(self):
        losses = self.update_th()
        mean_reward_player_row = -losses[0] * (1 - self.gamma)
        mean_reward_player_col = -losses[1] * (1 - self.gamma)
        to_log = {
            f"mean_reward_{self.players_ids[0]}": mean_reward_player_row,
            f"mean_reward_{self.players_ids[1]}": mean_reward_player_col,
            "episodes_total": self.training_iteration,
        }
        return to_log

    def _exact_loss_matrix_game_two_by_two_actions(self):
        pi_player_row_init_state = torch.sigmoid(
            self.weights_per_players[0][0:1]
        )
        pi_player_col_init_state = torch.sigmoid(
            self.weights_per_players[1][0:1]
        )
        p = torch.cat(
            [
                pi_player_row_init_state * pi_player_col_init_state,
                pi_player_row_init_state * (1 - pi_player_col_init_state),
                (1 - pi_player_row_init_state) * pi_player_col_init_state,
                (1 - pi_player_row_init_state)
                * (1 - pi_player_col_init_state),
            ]
        )
        pi_player_row_other_states = torch.reshape(
            torch.sigmoid(self.weights_per_players[0][1:5]), (4, 1)
        )
        pi_player_col_other_states = torch.reshape(
            torch.sigmoid(self.weights_per_players[1][1:5]), (4, 1)
        )
        P = torch.cat(
            [
                pi_player_row_other_states * pi_player_col_other_states,
                pi_player_row_other_states * (1 - pi_player_col_other_states),
                (1 - pi_player_row_other_states) * pi_player_col_other_states,
                (1 - pi_player_row_other_states)
                * (1 - pi_player_col_other_states),
            ],
            dim=1,
        )
        M = -torch.matmul(p, torch.inverse(torch.eye(4) - self.gamma * P))
        L_1 = torch.matmul(
            M, torch.reshape(self.payoff_matrix_player_row, (4, 1))
        )
        L_2 = torch.matmul(
            M, torch.reshape(self.payoff_matrix_player_col, (4, 1))
        )
        return [L_1, L_2]

    def init_weigths(self, std):
        self.weights_per_players = []
        for i in range(len(self.dims)):
            if std > 0:
                init = torch.nn.init.normal_(
                    torch.empty(self.dims[i], requires_grad=True), std=std
                )
            else:
                init = torch.zeros(self.dims[i], requires_grad=True)
            self.weights_per_players.append(init)

    def update_th(
        self,
        a=0.5,
        b=0.1,
        gam=1,
        ep=0.1,
        lss_lam=0.1,
    ):
        losses = self.exact_loss_function()

        grads = self._compute_gradients_wt_selected_method(
            a,
            b,
            ep,
            gam,
            losses,
            lss_lam,
        )

        self._update_weights(grads)
        return losses

    def _compute_gradients_wt_selected_method(
        self, a, b, ep, gam, losses, lss_lam
    ):
        n_states = len(self.weights_per_players)

        grad_L = _compute_vanilla_gradients(
            losses, n_states, self.weights_per_players
        )

        if self.method == "la":
            grads = _get_la_gradients(
                self.learning_rate, grad_L, n_states, self.weights_per_players
            )
        elif self.method == "lola":
            grads = _get_lola_exact_gradients(
                self.learning_rate, grad_L, n_states, self.weights_per_players
            )
        elif self.method == "sos":
            grads = _get_sos_gradients(
                a,
                self.learning_rate,
                b,
                grad_L,
                n_states,
                self.weights_per_players,
            )
        elif self.method == "sga":
            grads = _get_sga_gradients(
                ep, grad_L, n_states, self.weights_per_players
            )
        elif self.method == "co":
            grads = _get_co_gradients(
                gam, grad_L, n_states, self.weights_per_players
            )
        elif self.method == "eg":
            grads = _get_eg_gradients(
                self.exact_loss_function,
                self.learning_rate,
                losses,
                n_states,
                self.weights_per_players,
            )
        elif self.method == "cgd":  # Slow implementation (matrix inversion)
            grads = _get_cgd_gradients(
                self.learning_rate, grad_L, n_states, self.weights_per_players
            )
        elif self.method == "lss":  # Slow implementation (matrix inversion)
            grads = _get_lss_gradients(
                grad_L, lss_lam, n_states, self.weights_per_players
            )
        elif self.method == "naive":  # Naive Learning
            grads = _get_naive_learning_gradients(grad_L, n_states)
        else:
            raise ValueError(f"algo: {self.method}")
        return grads

    def _update_weights(self, grads):
        with torch.no_grad():
            for weight_i, weight_grad in enumerate(grads):
                self.weights_per_players[weight_i] -= (
                    self.learning_rate * weight_grad
                )

    def save_checkpoint(self, checkpoint_dir):
        save_path = os.path.join(checkpoint_dir, "weights.pt")
        torch.save(self.weights_per_players, save_path)
        return save_path

    def load_checkpoint(self, checkpoint_path):
        self.weights_per_players = torch.load(checkpoint_path)

    def _get_agent_to_use(self, policy_id):
        if policy_id == "player_row":
            agent_n = 0
        elif policy_id == "player_col":
            agent_n = 1
        else:
            raise ValueError(f"policy_id {policy_id}")
        return agent_n

    def _preprocess_obs(self, single_obs, agent_to_use):
        single_obs = np.where(single_obs == 1)[0][0]
        # because idx 0 is linked to the initial state in the weights
        # but this is the last idx which is linked to the
        # initial state in the environment obs
        if single_obs == len(self.weights_per_players[0]) - 1:
            single_obs = 0
        else:
            single_obs += 1
        return single_obs

    def _post_process_action(self, action):
        return action[None, ...]  # add batch dim

    def compute_actions(self, policy_id: str, obs_batch: list):
        assert (
            len(obs_batch) == 1
        ), f"{len(obs_batch)} == 1. obs_batch: {obs_batch}"

        for single_obs in obs_batch:
            agent_to_use = self._get_agent_to_use(policy_id)
            obs = self._preprocess_obs(single_obs, agent_to_use)
            policy = torch.sigmoid(self.weights_per_players[agent_to_use])
            coop_proba = policy[obs]
            if coop_proba > random.random():
                action = np.array(0)
            else:
                action = np.array(1)

        action = self._post_process_action(action)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches


def _get_la_gradients(alpha, grad_L, n, th):
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
        grad_L[i][i] - alpha * get_gradient(terms[i], th[i]) for i in range(n)
    ]
    return grads


def _get_lola_exact_gradients(alpha, grad_L, n, th):
    terms = [
        sum(
            [torch.dot(grad_L[j][i], grad_L[j][j]) for j in range(n) if j != i]
        )
        for i in range(n)
    ]
    grads = [
        grad_L[i][i] - alpha * get_gradient(terms[i], th[i]) for i in range(n)
    ]
    return grads


def _get_sos_gradients(a, alpha, b, grad_L, n, th):
    xi_0 = _get_la_gradients(alpha, grad_L, n, th)
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
    p1 = 1 if dot >= 0 else min(1, -a * torch.norm(torch.cat(xi_0)) ** 2 / dot)
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    xi_norm = torch.norm(xi)
    p2 = xi_norm ** 2 if xi_norm < b else 1
    p = min(p1, p2)
    grads = [xi_0[i] - p * alpha * chi[i] for i in range(n)]
    return grads


def _get_sga_gradients(ep, grad_L, n, th):
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
    return grads


def _get_co_gradients(gam, grad_L, n, th):
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    ham = torch.dot(xi, xi.detach())
    grads = [grad_L[i][i] + gam * get_gradient(ham, th[i]) for i in range(n)]
    return grads


def _get_eg_gradients(Ls, alpha, losses, n, th):
    th_eg = [th[i] - alpha * get_gradient(losses[i], th[i]) for i in range(n)]
    losses_eg = Ls(th_eg)
    grads = [get_gradient(losses_eg[i], th_eg[i]) for i in range(n)]
    return grads


def _get_naive_learning_gradients(grad_L, n):
    grads = [grad_L[i][i] for i in range(n)]
    return grads


def _get_lss_gradients(grad_L, lss_lam, n, th):
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
        torch.matmul(torch.eye(sum(dims)) + torch.matmul(H.T, H_inv), xi) / 2
    )
    grads = [grad[sum(dims[:i]) : sum(dims[: i + 1])] for i in range(n)]
    return grads


def _get_cgd_gradients(alpha, grad_L, n, th):
    dims = [len(th[i]) for i in range(n)]
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    H_o = get_hessian(th, grad_L, diag=False)
    grad = torch.matmul(torch.inverse(torch.eye(sum(dims)) + alpha * H_o), xi)
    grads = [grad[sum(dims[:i]) : sum(dims[: i + 1])] for i in range(n)]
    return grads


def _compute_vanilla_gradients(losses, n, th):
    grad_L = [
        [get_gradient(losses[j], th[i]) for j in range(n)] for i in range(n)
    ]
    return grad_L


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
                        get_gradient(grad_L[i][i][k], th[j]),
                        dim=0,
                    )
                    for k in range(len(th[i]))
                ]
                row_block.append(torch.cat(block, dim=0))
            else:
                row_block.append(torch.zeros(len(th[i]), len(th[j])))
        H.append(torch.cat(row_block, dim=1))
    return torch.cat(H, dim=0)
