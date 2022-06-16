######
# Code modified from:
# https://github.com/julianstastny/openspiel-social-dilemmas
######
import logging
import os
import random
from typing import Iterable

import numpy as np
import torch
from marltoolbox.algos import algo_globals
from marltoolbox.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
    IteratedAsymBoS,
    IteratedThreatGame,
)
from ray import tune

logger = logging.getLogger(__name__)

THREAT_GAME_STATE_ORDER = [
    "Init",
    "G+TR",
    "G+TS",
    "G+NT",
    "NG+TR",
    "NG+TS",
    "NG+NT",
]
GRAD_MUL = 1.0
XI_MUL = 1.0
PL1_LR_SCALING = 1.0


class SOSTrainer(tune.Trainable):
    def setup(self, config: dict):

        if config:
            self.config = config
            self.gamma = self.config.get("gamma")
            self.learning_rate = self.config.get("lr")
            self.method = self.config.get("method")
            self._inner_epochs = self.config["inner_epochs"]
            self.use_single_weights = False
            self.seed = self.config["seed"]
            self._meta_learn_reward_fn = self.config.get("meta_learn_reward_fn", False)
            self._relatif_reward_model = self.config.get("relatif_reward_model", None)
            self._convert_log_to_numpy = self.config.get("convert_log_to_numpy", True)
            self._lr_warmup = self.config.get("lr_warmup_inner", None)
            self._momentum = self.config.get("momentum_inner", 0.0)
            self._last_grads = [0.0, 0.0]
            self._n_updates_done = 0

            global GRAD_MUL, XI_MUL, PL1_LR_SCALING
            XI_MUL = self.config.get("xi_mul", 1.0)
            GRAD_MUL = self.config.get("grad_mul", 1.0)
            PL1_LR_SCALING = self.config.get("pl1_lr_scaling", 1.0)

            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            self._set_environment()
            self.init_weigths(std=self.config.get("inital_weights_std"))
            self._use_naive_grad = self.config.get(
                "use_naive_grad", [False] * self.n_players
            )
            if self.use_single_weights:
                self._exact_loss_matrix_game = (
                    self._exact_loss_matrix_game_two_by_two_actions
                )
            else:
                self._exact_loss_matrix_game = self._exact_loss_matrix_game_generic

    def _set_environment(self):

        self.players_ids, self._state_order, payoff_matrix = get_payoff_matrix(
            self.config
        )

        if not isinstance(payoff_matrix, torch.Tensor):
            payoff_matrix = torch.tensor(payoff_matrix)

        payoff_matrix = payoff_matrix.float()

        self.payoff_matrix_player_row = payoff_matrix[:, :, 0]
        self.payoff_matrix_player_col = payoff_matrix[:, :, 1]

        self.n_actions_p1 = payoff_matrix.shape[0]
        self.n_actions_p2 = payoff_matrix.shape[1]
        self.n_non_init_states = self.n_actions_p1 * self.n_actions_p2

        if self.use_single_weights:
            self.dims = [
                (self.n_non_init_states + 1,),
                (self.n_non_init_states + 1,),
            ]
        else:
            self.dims = [
                (self.n_non_init_states + 1, self.n_actions_p1),
                (self.n_non_init_states + 1, self.n_actions_p2),
            ]

        # Can be used to observe the order of the States associated with the weights
        # torch.reshape(self.payoff_matrix_player_row, (self.n_non_init_states, 1))

    def step(self):
        for _ in range(self._inner_epochs):
            losses = self.update_th()

        mean_reward_player_row = -losses[0] * (1 - self.gamma)
        mean_reward_player_col = -losses[1] * (1 - self.gamma)
        current_pl_weights = (
            self.weights_per_players[self._inner_epoch_idx]
            if self._meta_learn_reward_fn
            else self.weights_per_players
        )
        to_log = {
            f"mean_reward_{self.players_ids[0]}": float(mean_reward_player_row),
            f"mean_reward_{self.players_ids[1]}": float(mean_reward_player_col),
            "episodes_total": self.training_iteration,
            "weights_per_players": current_pl_weights
            if self._meta_learn_reward_fn
            else [p_weights.tolist() for p_weights in current_pl_weights],
        }

        for state_idx, proba_all_a in enumerate(self.policy_player1.tolist()):
            state = self._state_order[state_idx]
            to_log[format_pl_s_a("1", state)] = list(proba_all_a)
            for act_idx, act_proba in enumerate(proba_all_a):
                to_log[format_pl_s_a("1", state, act_idx)] = act_proba
        for state_idx, proba_all_a in enumerate(self.policy_player2.tolist()):
            state = self._state_order[state_idx]
            to_log[format_pl_s_a("2", state)] = list(proba_all_a)
            for act_idx, act_proba in enumerate(proba_all_a):
                to_log[format_pl_s_a("2", state, act_idx)] = act_proba
        to_log[f"P(S)"] = self.proba_states.tolist()
        for state_idx, proba_s in enumerate(self.proba_states.tolist()):
            state = self._state_order[state_idx]
            to_log[f"P(s={state})"] = proba_s

        if self._convert_log_to_numpy:
            to_log = convert_to_float(to_log)
        return to_log

    def _exact_loss_matrix_game_two_by_two_actions(self):

        if self._meta_learn_reward_fn:
            raise NotImplementedError()

        self.policy_player1 = torch.sigmoid(self.weights_per_players[0])
        self.policy_player2 = torch.sigmoid(self.weights_per_players[1])

        pi_player_row_init_state = torch.sigmoid(self.weights_per_players[0][0:1])
        pi_player_col_init_state = torch.sigmoid(self.weights_per_players[1][0:1])
        p = torch.cat(
            [
                pi_player_row_init_state * pi_player_col_init_state,
                pi_player_row_init_state * (1 - pi_player_col_init_state),
                (1 - pi_player_row_init_state) * pi_player_col_init_state,
                (1 - pi_player_row_init_state) * (1 - pi_player_col_init_state),
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
                (1 - pi_player_row_other_states) * (1 - pi_player_col_other_states),
            ],
            dim=1,
        )
        M = -torch.matmul(p, torch.inverse(torch.eye(4) - self.gamma * P))
        L_1 = torch.matmul(M, torch.reshape(self.payoff_matrix_player_row, (4, 1)))
        L_2 = torch.matmul(M, torch.reshape(self.payoff_matrix_player_col, (4, 1)))
        return [L_1, L_2]

    def _exact_loss_matrix_game_generic(self):

        if self._meta_learn_reward_fn:
            pl1_weights = self.weights_per_players[self._inner_epoch_idx][0]
            pl2_weights = self.weights_per_players[self._inner_epoch_idx][1]
        else:
            pl1_weights = self.weights_per_players[0]
            pl2_weights = self.weights_per_players[1]

        pi_player_row = torch.sigmoid(pl1_weights)
        pi_player_col = torch.sigmoid(pl2_weights)
        sum_1 = torch.sum(pi_player_row, dim=1)
        sum_1 = torch.stack([sum_1 for _ in range(self.n_actions_p1)], dim=1)
        sum_2 = torch.sum(pi_player_col, dim=1)
        sum_2 = torch.stack([sum_2 for _ in range(self.n_actions_p2)], dim=1)
        pi_player_row = pi_player_row / sum_1
        pi_player_col = pi_player_col / sum_2
        # The softmax could be used on dim=1 but then the probabilities are
        # no more the same and the std used to generate the weights should be reduced
        # pi_player_row_bis = torch.nn.functional.softmax(pl1_weights, dim=1)
        # pi_player_col_bis = torch.nn.functional.softmax(pl2_weights, dim=1)
        # assert torch.allclose(pi_player_row, pi_player_row_bis)
        # assert torch.allclose(pi_player_col, pi_player_col_bis)
        self.policy_player1 = pi_player_row
        self.policy_player2 = pi_player_col
        # idx 0 in the weight is link to the initial state
        # pi_player_row_init_state = pi_player_row[:1, :]
        # pi_player_col_init_state = pi_player_col[:1, :]
        # all_initial_actions_proba_pairs = []
        # for action_p1 in range(self.n_actions_p1):
        #     for action_p2 in range(self.n_actions_p2):
        #         all_initial_actions_proba_pairs.append(
        #             pi_player_row_init_state[:, action_p1]
        #             * pi_player_col_init_state[:, action_p2]
        #         )
        # p = torch.cat(
        #     all_initial_actions_proba_pairs,
        # )
        p = torch.matmul(pi_player_row[:1, :].T, pi_player_col[:1, :]).flatten()
        # assert torch.allclose(p, pbis)

        # pi_player_row_other_states = pi_player_row[1:, :]
        # pi_player_col_other_states = pi_player_col[1:, :]
        # all_actions_proba_pairs = []
        # for action_p1 in range(self.n_actions_p1):
        #     for action_p2 in range(self.n_actions_p2):
        #         all_actions_proba_pairs.append(
        #             pi_player_row_other_states[:, action_p1]
        #             * pi_player_col_other_states[:, action_p2]
        #         )
        # P = torch.stack(
        #     all_actions_proba_pairs,
        #     1,
        # )
        # Pbis = torch.matmul(pi_player_col[1:, :], pi_player_row[1:, :].T)
        P = torch.stack(
            [
                torch.matmul(
                    pi_player_row[i, :].unsqueeze(dim=-1),
                    pi_player_col[i, :].unsqueeze(dim=0),
                ).flatten()
                for i in range(1, 7)
            ],
            dim=0,
        )
        # assert torch.allclose(P, Pter)

        # Probabilities of states for an infinite episode with gamma as the discound factor
        M = -torch.matmul(
            p,
            torch.inverse(torch.eye(self.n_non_init_states) - self.gamma * P),
        )
        self.proba_states = -M * (1 - self.gamma)
        L_1 = torch.matmul(
            M,
            torch.reshape(self.payoff_matrix_player_row, (self.n_non_init_states, 1)),
        )
        L_2 = torch.matmul(
            M,
            torch.reshape(self.payoff_matrix_player_col, (self.n_non_init_states, 1)),
        )

        # Plot the grad graph
        # from torchviz import make_dot
        #
        # dot = make_dot(
        #     L_1,
        #     params={"pl1_weights": pl1_weights},
        #     # show_attrs=True,
        #     # show_saved=True,
        # )
        # filename = os.path.join(os.getcwd(), "meta_gradient_reward_fn")
        # dot.render(filename=filename, format="png")
        # assert 0
        # dot = make_dot(
        #     L_2,
        #     params={"pl2_weights": pl2_weights},
        #     # show_attrs=True,
        #     # show_saved=True,
        # )

        return [L_1, L_2]

    def init_weigths(self, std):
        self.weights_per_players = []
        self.n_players = len(self.dims)
        for i in range(self.n_players):
            if std > 0:
                init = torch.nn.init.normal_(
                    torch.empty(self.dims[i], requires_grad=True), std=std
                )
            else:
                init = torch.zeros(self.dims[i], requires_grad=True)
            self.weights_per_players.append(init)

        if self._meta_learn_reward_fn:
            self.weights_per_players = [self.weights_per_players]
            self._inner_epoch_idx = 0

    def update_th(
        self,
        a=0.5,
        b=0.1,
        gam=1,
        ep=0.1,
        lss_lam=0.1,
    ):
        losses = self._exact_loss_matrix_game()

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

    def _compute_gradients_wt_selected_method(self, a, b, ep, gam, losses, lss_lam):
        n_players = self.n_players

        if self._meta_learn_reward_fn:
            weights_per_players = self.weights_per_players[self._inner_epoch_idx]
        else:
            weights_per_players = self.weights_per_players

        grad_L = _compute_vanilla_gradients(losses, n_players, weights_per_players)

        if self.method == "la":
            grads = _get_la_gradients(
                self.learning_rate, grad_L, n_players, weights_per_players
            )
        elif self.method == "lola":
            grads = _get_lola_exact_gradients(
                self.learning_rate, grad_L, n_players, weights_per_players
            )
        elif self.method == "sos":
            grads = _get_sos_gradients(
                a,
                self.learning_rate,
                b,
                grad_L,
                n_players,
                weights_per_players,
            )
        elif self.method == "sga":
            grads = _get_sga_gradients(ep, grad_L, n_players, weights_per_players)
        elif self.method == "co":
            grads = _get_co_gradients(gam, grad_L, n_players, weights_per_players)
        elif self.method == "eg":
            grads = _get_eg_gradients(
                self._exact_loss_matrix_game,
                self.learning_rate,
                losses,
                n_players,
                weights_per_players,
            )
        elif self.method == "cgd":  # Slow implementation (matrix inversion)
            grads = _get_cgd_gradients(
                self.learning_rate, grad_L, n_players, weights_per_players
            )
        elif self.method == "lss":  # Slow implementation (matrix inversion)
            grads = _get_lss_gradients(grad_L, lss_lam, n_players, weights_per_players)
        elif self.method == "naive":  # Naive Learning
            grads = (grad_L, n_players)
        else:
            raise ValueError(f"algo: {self.method}")

        if any(self._use_naive_grad):
            # naive_grads = (grad_L, n_players)
            grads = [
                pl_i_naive_grad[idx] * GRAD_MUL if pl_i_use_naive_grad else pl_i_grad
                for idx, (pl_i_use_naive_grad, pl_i_grad, pl_i_naive_grad) in enumerate(
                    zip(self._use_naive_grad, grads, grad_L)
                )
            ]

        return grads

    def _update_weights(self, grads):
        if self._meta_learn_reward_fn:
            self.weights_per_players.append([None] * len(grads))
            for weight_i, weight_grad in enumerate(grads):
                if weight_i == 0:
                    weight_grad *= PL1_LR_SCALING
                if self._momentum:
                    weight_grad = (
                        weight_grad + self._momentum * self._last_grads[weight_i]
                    )
                    self._last_grads[weight_i] = weight_grad
                learning_rate = self.learning_rate
                if self._lr_warmup and self._n_updates_done < self._lr_warmup:
                    learning_rate = (
                        learning_rate * self._n_updates_done / self._lr_warmup
                    )
                self.weights_per_players[self._inner_epoch_idx + 1][
                    weight_i
                ] = self.weights_per_players[self._inner_epoch_idx][weight_i] - (
                    learning_rate * weight_grad / GRAD_MUL
                )
            self._inner_epoch_idx += 1
        else:
            with torch.no_grad():
                for weight_i, weight_grad in enumerate(grads):
                    if weight_i == 0:
                        weight_grad *= PL1_LR_SCALING
                    if self._momentum:
                        weight_grad = (
                            weight_grad + self._momentum * self._last_grads[weight_i]
                        )
                        self._last_grads[weight_i] = weight_grad
                    learning_rate = self.learning_rate
                    if self._lr_warmup and self._n_updates_done < self._lr_warmup:
                        learning_rate = (
                            learning_rate * self._n_updates_done / self._lr_warmup
                        )
                    self.weights_per_players[weight_i] -= (
                        learning_rate * weight_grad / GRAD_MUL
                    )
        self._n_updates_done += 1

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
        assert len(obs_batch) == 1, f"{len(obs_batch)} == 1. obs_batch: {obs_batch}"

        for single_obs in obs_batch:
            agent_to_use = self._get_agent_to_use(policy_id)
            obs = self._preprocess_obs(single_obs, agent_to_use)
            policy = torch.sigmoid(self.weights_per_players[agent_to_use])

            if self.use_single_weights:
                coop_proba = policy[obs]
                if coop_proba > random.random():
                    action = np.array(0)
                else:
                    action = np.array(1)
            else:
                probabilities = policy[obs, :]
                probabilities = torch.tensor(probabilities)
                policy_for_this_state = torch.distributions.Categorical(
                    probs=probabilities
                )
                action = np.array(policy_for_this_state.sample())

        action = self._post_process_action(action)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches


def _compute_vanilla_gradients(losses, n, th):
    grad_L = [[get_gradient(losses[j], th[i]) for j in range(n)] for i in range(n)]
    return grad_L


def _get_la_gradients(alpha, grad_L, n, th):
    terms = [
        sum(
            [
                torch.matmul(
                    grad_L[j][i],
                    torch.transpose(grad_L[j][j].detach(), 0, 1),
                )
                for j in range(n)
                if j != i
            ]
        )
        for i in range(n)
    ]
    grads = [grad_L[i][i] - alpha * get_gradient(terms[i], th[i]) for i in range(n)]
    return grads


def _get_lola_exact_gradients(alpha, grad_L, n, th):
    terms = [
        sum(
            [
                torch.matmul(
                    grad_L[j][i],
                    torch.transpose(grad_L[j][j].unsqueeze(dim=1), 0, 1),
                )
                for j in range(n)
                if j != i
            ]
        )
        for i in range(n)
    ]
    grads = [grad_L[i][i] - alpha * get_gradient(terms[i], th[i]) for i in range(n)]
    return grads


def _get_sos_gradients(a, alpha, b, grad_L, n, th):

    # Make it work for non symmetrical actions spaces (unsqueeze)
    n_actions = [[el_bis.shape[1] for el_bis in el] for el in grad_L]
    max_n_actions = int(np.max(n_actions))
    grad_L_reshaped = [
        [
            torch.cat(
                [
                    el_bis,
                    torch.zeros(
                        (el_bis.shape[0], max_n_actions - el_bis.shape[1]),
                        requires_grad=True,
                    ),
                ],
                dim=1,
            )
            # el_bis
            if el_bis.shape[1] < max_n_actions else el_bis
            for el_bis in el
        ]
        for el in grad_L
    ]

    xi_0 = _get_la_gradients(alpha, grad_L, n, th)
    chi = [
        get_gradient(
            sum(
                [
                    torch.matmul(
                        grad_L[j][i].detach(),
                        torch.transpose(grad_L[j][j], 0, 1),
                    )
                    for j in range(n)
                    if j != i
                ]
            ),
            th[i],
        )
        for i in range(n)
    ]

    # Make it work for different actions spaces
    # chi_n_actions = [el.shape[1] for el in chi]
    # max_n_actions = max(chi_n_actions)
    chi_reshaped = [
        torch.cat([el, torch.zeros((el.shape[0], max_n_actions - el.shape[1]))], dim=1)
        # el
        if el.shape[1] < max_n_actions else el
        for el in chi
    ]
    xi_0_reshaped = [
        torch.cat([el, torch.zeros((el.shape[0], max_n_actions - el.shape[1]))], dim=1)
        # el
        if el.shape[1] < max_n_actions else el
        for el in xi_0
    ]

    # Compute p
    dot = torch.matmul(
        -alpha * torch.cat(xi_0_reshaped),
        torch.transpose(torch.cat(xi_0_reshaped), 0, 1),
    ).sum()
    p1 = 1 if dot >= 0 else min(1, -a * torch.norm(torch.cat(xi_0_reshaped)) ** 2 / dot)
    xi = torch.cat([grad_L_reshaped[i][i] for i in range(n)])
    # Balance making gamma = 0.0
    xi = xi * XI_MUL
    xi_norm = torch.norm(xi)
    p2 = xi_norm**2 if xi_norm < b else 1
    p = min(p1, p2)
    grads = [xi_0_reshaped[i] - p * alpha * chi_reshaped[i] for i in range(n)]

    # Make it work for different actions spaces (squeeze)
    grads = [
        grads_pl_i[tuple(slice(d) for d in th_pl_i.shape)]
        if th_pl_i.shape != grads_pl_i.shape
        else grads_pl_i
        for grads_pl_i, th_pl_i in zip(grads, th)
    ]

    return grads


def _get_sga_gradients(ep, grad_L, n, th):
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    ham = torch.dot(xi, xi.detach())
    H_t_xi = [get_gradient(ham, th[i]) for i in range(n)]
    H_xi = [
        get_gradient(
            sum([torch.dot(grad_L[j][i], grad_L[j][j].detach()) for j in range(n)]),
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
        inv = torch.inverse(torch.matmul(H.T, H) + lss_lam * torch.eye(sum(dims)))
        H_inv = torch.matmul(inv, H.T)
    else:
        H_inv = torch.inverse(H)
    grad = torch.matmul(torch.eye(sum(dims)) + torch.matmul(H.T, H_inv), xi) / 2
    grads = [grad[sum(dims[:i]) : sum(dims[: i + 1])] for i in range(n)]
    return grads


def _get_cgd_gradients(alpha, grad_L, n, th):
    dims = [len(th[i]) for i in range(n)]
    xi = torch.cat([grad_L[i][i] for i in range(n)])
    H_o = get_hessian(th, grad_L, diag=False)
    grad = torch.matmul(torch.inverse(torch.eye(sum(dims)) + alpha * H_o), xi)
    grads = [grad[sum(dims[:i]) : sum(dims[: i + 1])] for i in range(n)]
    return grads


def get_gradient(function, param):
    grad = torch.autograd.grad(
        torch.sum(function), param, create_graph=True  # , allow_unused=True
    )[0]
    # to balance setting gamma to 0.0
    grad = grad * GRAD_MUL
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


def format_pl_s_a(pl, s, a=None):
    if a is None:
        return f"pl{pl}_s{s}"
    return f"pl{pl}_s{s}_a{a}"


def convert_to_float(to_log):
    if isinstance(to_log, dict):
        for k, v in list(to_log.items()):
            if isinstance(v, torch.Tensor):
                to_log[k] = to_log[k].detach().numpy().tolist()
                to_log = unnest_list(to_log, k)
            elif isinstance(v, list) or isinstance(v, tuple):
                for i in range(len(v)):
                    new_k = f"{k}_{i}"
                    to_log[new_k] = convert_to_float(v[i])
                    to_log = unnest_list(to_log, new_k)
    elif isinstance(to_log, list) or isinstance(to_log, tuple):
        to_log = [convert_to_float(el) for el in to_log]
    elif isinstance(to_log, torch.Tensor):
        to_log = to_log.detach().numpy().tolist()
        # if isinstance(to_log, list):
        #     to_log = to_log[0]
    return to_log


def unnest_list(to_log, k):
    if isinstance(to_log[k], Iterable):
        for i in range(len(to_log[k])):
            new_k = f"{k}_{i}"
            to_log[new_k] = to_log[k][i]
            if isinstance(to_log[new_k], Iterable):
                to_log = unnest_list(to_log, new_k)
    return to_log


def get_payoff_matrix(config):
    state_order = None
    if "custom_payoff_threat_game" in config.keys():
        if (
            isinstance(config["custom_payoff_threat_game"], str)
            and config["custom_payoff_threat_game"] == "use_global"
        ):
            payoff_matrix = algo_globals.MODIFIED_PAYOFF_MATRIX
            print(f"Use global var to set the payoff matrix: {payoff_matrix.tolist()}")
        else:
            payoff_matrix = config["custom_payoff_threat_game"]
            print(f"Use a custom threat game: {payoff_matrix.tolist()}")
        env_class = IteratedThreatGame
        players_ids = env_class({}).players_ids
        state_order = THREAT_GAME_STATE_ORDER
    elif "custom_payoff_matrix" in config.keys():
        payoff_matrix = config["custom_payoff_matrix"]
        players_ids = IteratedPrisonersDilemma({}).players_ids
        print(f"Use a custom game: {np.array(payoff_matrix).tolist()}")
    elif config.get("env_name") == "IteratedPrisonersDilemma":
        env_class = IteratedPrisonersDilemma
        payoff_matrix = env_class.PAYOFF_MATRIX
        players_ids = env_class({}).players_ids
        state_order = [
            "Init",
            "CC",
            "CD",
            "DC",
            "DD",
        ]
    elif config.get("env_name") == "IteratedAsymBoS":
        env_class = IteratedAsymBoS
        payoff_matrix = env_class.PAYOFF_MATRIX
        players_ids = env_class({}).players_ids
        state_order = [
            "Init",
            "CC",
            "CD",
            "DC",
            "DD",
        ]
    elif config.get("env_name") == "IteratedThreatGame":
        env_class = IteratedThreatGame
        payoff_matrix = env_class.PAYOFF_MATRIX
        players_ids = env_class({}).players_ids
        state_order = THREAT_GAME_STATE_ORDER

    else:
        raise NotImplementedError()

    return players_ids, state_order, payoff_matrix
