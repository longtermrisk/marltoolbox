##########
# Part of the code modified from:
# https://github.com/asappresearch/emergent-comms-negotiation
##########

import argparse
import datetime
from ray import tune
import os

from marltoolbox.algos.alternating_offers.alt_offers_training import (
    AltOffersTraining,
)
from marltoolbox.utils import miscellaneous, log


def main():
    config = get_config()
    print(config)

    if config["enable_cuda"]:
        raise NotImplementedError

    # each logical step of training contains several episodes, each episode is a batch of games
    training_steps = config["training_episodes"] // config["episodes_per_step"]
    print(f"Num of training steps: {training_steps}")
    print(f'Episodes per step: {config["episodes_per_step"]}')

    exp_name_expanded, exp_dir = log.log_in_current_day_dir(config["name"])

    analysis = tune.run(
        name=exp_name_expanded,
        run_or_experiment=AltOffersTraining,
        stop={"training_iteration": training_steps},
        config=config,
        checkpoint_freq=1000,
        checkpoint_at_end=True,
        metric="prosocial_reward",
        mode="max",
    )

    #     check_learning_achieved(tune_results=analysis, metric='prosocial_reward')
    log.save_metrics(analysis, exp_name_expanded, "metrics.pickle")
    log.pprint_saved_metrics(
        os.path.join(
            os.path.expanduser("~/ray_results"),
            exp_name_expanded,
            "metrics.pickle",
        )
    )
    best_checkpoints = miscellaneous.extract_checkpoints(analysis)
    print(best_checkpoints)


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", type=str, default="default", help="used for logfiles naming"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128
    )  # size of one batch = size of one episode in number of games
    parser.add_argument(
        "--training_episodes", type=int, default=100000
    )  # number of batches processed in the course of training

    parser.add_argument(
        "--enable_binding_comm", action="store_true"
    )  # enable the exchange of binding proposals
    parser.add_argument(
        "--enable_cheap_comm", action="store_true"
    )  # enable the exchange of cheap utterances
    parser.add_argument(
        "--hidden_embedding_sizes", type=str, default="30,30"
    )  # width of hidden layers of agents
    parser.add_argument(
        "--utility_types", type=str, default="uniform,uniform"
    )  # types of the distribution over utility functions for player 1 and player 2
    # proportion of the final reward of agents that accounts for the sum of raw rewards of both agents, can range from 0 (selfish agents) to 1 (cooperation)
    parser.add_argument("--prosociality_levels", type=str, default="0,0")
    parser.add_argument("--fairness_coeffs", type=str, default="0,0")

    parser.add_argument(
        "--enable_arbitrator", action="store_true"
    )  # enable the arbitrator training
    parser.add_argument(
        "--scale_before_redist", action="store_true"
    )  # if normalizing the rewards happens before or after they are redistributed by the arbitrator

    parser.add_argument(
        "--response_entropy_reg", type=float
    )  # regularization coeff for entropy of action distribution
    parser.add_argument(
        "--utterance_entropy_reg", type=float
    )  # regularization coeff for entropy of utterance distribution
    parser.add_argument(
        "--proposal_entropy_reg", type=float
    )  # regularization coeff for entropy of binding proposal distribution
    parser.add_argument(
        "--arbitrator_entropy_reg", type=float
    )  # regularization coeff for entropy of arbitrator decisions
    parser.add_argument(
        "--arbitrator_main_loss_coeff", type=float
    )  # coeff that balances between arbitrator loss and agents' losses

    #     parser.add_argument('--test_seed', type=int, default=123)  # this was for testing mode with deterministic argmax actions, disabled it for now
    parser.add_argument("--train_seed", type=int, help="optional")
    parser.add_argument("--enable_cuda", action="store_true")
    parser.add_argument(
        "--episodes_per_step", type=int, default=64
    )  # how many episodes are in a Tune step()
    parser.add_argument("--suppress_output", action="store_true")

    parser.add_argument(
        "--agents_sgd", action="store_true"
    )  # use SGD for training of 2 player agents instead of Adam
    parser.add_argument(
        "--arbitrator_sgd", action="store_true"
    )  # use SGD for training of the arbitrator instead of Adam
    parser.add_argument(
        "--share_utilities", action="store_true"
    )  # whether agents share their utility functions with the arbitrator
    parser.add_argument(
        "--enable_overflow", action="store_true"
    )  # variant of the game where items are destroyed if an agreement is not reached for a type of item

    args = parser.parse_args()
    args = args.__dict__
    args["hidden_embedding_sizes"] = [
        int(param) for param in args["hidden_embedding_sizes"].split(",")
    ]
    args["utility_types"] = args["utility_types"].split(",")
    args["prosociality_levels"] = [
        float(param) for param in args["prosociality_levels"].split(",")
    ]
    args["fairness_coeffs"] = [
        float(param) for param in args["fairness_coeffs"].split(",")
    ]
    return args


if __name__ == "__main__":
    main()
