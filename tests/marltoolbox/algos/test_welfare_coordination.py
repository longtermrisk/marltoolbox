import random
from marltoolbox.algos.welfare_coordination import MetaGameSolver

BEST_PAYOFF = 1.5
BEST_WELFARE = "best_welfare"
WORST_PAYOFF = -0.5
WORST_WELFARE = "worst_welfare"


def test_end_to_end_wt_best_welfare_fn():
    for _ in range(10):
        meta_game_solver = _given_meta_game_with_a_clear_extrem_welfare_fn(
            best=True
        )
        for _ in range(10):
            _when_solving_meta_game(meta_game_solver)
            _assert_best_welfare_in_announced_set(meta_game_solver)


def test_end_to_end_wt_worst_welfare_fn():
    for _ in range(10):
        meta_game_solver = _given_meta_game_with_a_clear_extrem_welfare_fn(
            worst=True
        )
        for _ in range(10):
            _when_solving_meta_game(meta_game_solver)
            _assert_best_welfare_not_in_announced_set(meta_game_solver)


def _given_meta_game_with_a_clear_extrem_welfare_fn(best=False, worst=False):
    assert best or worst
    assert not (best and worst)
    meta_game_solver = MetaGameSolver()
    n_welfares = _get_random_number_of_welfare_fn()
    own_player_idx = _get_random_position_of_players()
    opp_player_idx = (own_player_idx + 1) % 2
    welfares = ["welfare_" + str(el) for el in list(range(n_welfares - 1))]
    if best:
        welfares.append(BEST_WELFARE)
    elif worst:
        welfares.append(WORST_WELFARE)

    all_welfare_pairs_wt_payoffs = (
        _get_all_welfare_pairs_wt_extrem_payoffs_for_i(
            welfares=welfares,
            own_player_idx=own_player_idx,
            best_welfare=BEST_WELFARE if best else None,
            worst_welfare=WORST_WELFARE if worst else None,
        )
    )

    meta_game_solver.setup_meta_game(
        all_welfare_pairs_wt_payoffs,
        own_player_idx=own_player_idx,
        opp_player_idx=opp_player_idx,
        own_default_welfare_fn=welfares[own_player_idx],
        opp_default_welfare_fn=welfares[opp_player_idx],
    )
    return meta_game_solver


def _when_solving_meta_game(meta_game_solver):
    meta_game_solver.solve_meta_game(_get_random_tau())


def _assert_best_welfare_in_announced_set(meta_game_solver):
    assert BEST_WELFARE in meta_game_solver.welfare_set_to_annonce


def _assert_best_welfare_not_in_announced_set(meta_game_solver):
    assert BEST_WELFARE not in meta_game_solver.welfare_set_to_annonce


def _get_random_tau():
    return random.random()


def _get_random_number_of_welfare_fn():
    return random.randint(2, 4)


def _get_random_position_of_players():
    return random.randint(0, 1)


def _get_all_welfare_pairs_wt_extrem_payoffs_for_i(
    welfares,
    own_player_idx,
    best_welfare: str = None,
    worst_welfare: str = None,
):
    all_welfare_pairs_wt_payoffs = {}
    for welfare_p1 in welfares:
        for welfare_p2 in welfares:
            welfare_pair_name = (
                MetaGameSolver.from_pair_of_welfare_names_to_key(
                    welfare_p1, welfare_p2
                )
            )

            all_welfare_pairs_wt_payoffs[welfare_pair_name] = [
                random.random(),
                random.random(),
            ]
            if best_welfare is not None and best_welfare == welfare_p1:
                all_welfare_pairs_wt_payoffs[welfare_pair_name][
                    own_player_idx
                ] = BEST_PAYOFF
            elif worst_welfare is not None and worst_welfare == welfare_p1:
                all_welfare_pairs_wt_payoffs[welfare_pair_name][
                    own_player_idx
                ] = WORST_PAYOFF
    return all_welfare_pairs_wt_payoffs


def test__compute_meta_payoff():
    for _ in range(100):
        (
            welfares,
            all_welfare_pairs_wt_payoffs,
            own_welfare_set,
            opp_welfare_set,
            payoff,
            payoff_default,
            own_default_welfare_fn,
            opp_default_welfare_fn,
            own_player_idx,
            opp_player_idx,
        ) = _given_this_all_welfare_pairs_wt_payoffs()

        meta_payoff = _when_computing_meta_game_payoff(
            all_welfare_pairs_wt_payoffs,
            own_player_idx,
            opp_player_idx,
            own_default_welfare_fn,
            opp_default_welfare_fn,
            own_welfare_set,
            opp_welfare_set,
        )

        _assert_get_the_right_payoffs_or_default_payoff(
            own_welfare_set,
            opp_welfare_set,
            own_player_idx,
            meta_payoff,
            payoff,
            payoff_default,
        )


def _when_computing_meta_game_payoff(
    all_welfare_pairs_wt_payoffs,
    own_player_idx,
    opp_player_idx,
    own_default_welfare_fn,
    opp_default_welfare_fn,
    own_welfare_set,
    opp_welfare_set,
):
    meta_game_solver = MetaGameSolver()
    meta_game_solver.setup_meta_game(
        all_welfare_pairs_wt_payoffs,
        own_player_idx=own_player_idx,
        opp_player_idx=opp_player_idx,
        own_default_welfare_fn=own_default_welfare_fn,
        opp_default_welfare_fn=opp_default_welfare_fn,
    )
    meta_payoff = meta_game_solver._compute_meta_payoff(
        own_welfare_set, opp_welfare_set
    )
    return meta_payoff


def _given_this_all_welfare_pairs_wt_payoffs():
    n_welfares = _get_random_number_of_welfare_fn()
    welfares = ["welfare_" + str(el) for el in list(range(n_welfares))]
    own_player_idx = _get_random_position_of_players()
    opp_player_idx = (own_player_idx + 1) % 2
    all_welfare_pairs_wt_payoffs = {}

    own_welfare_set, opp_welfare_set, payoff = _add_nominal_case(
        welfares, all_welfare_pairs_wt_payoffs, own_player_idx
    )

    (
        own_default_welfare_fn,
        opp_default_welfare_fn,
        payoff_default,
    ) = _add_default_case(
        welfares, all_welfare_pairs_wt_payoffs, own_player_idx
    )

    if (
        own_default_welfare_fn
        == list(own_welfare_set)[0]
        == list(opp_welfare_set)[0]
        == opp_default_welfare_fn
    ):
        payoff = payoff_default
    return (
        welfares,
        all_welfare_pairs_wt_payoffs,
        own_welfare_set,
        opp_welfare_set,
        payoff,
        payoff_default,
        own_default_welfare_fn,
        opp_default_welfare_fn,
        own_player_idx,
        opp_player_idx,
    )


def _assert_get_the_right_payoffs_or_default_payoff(
    own_welfare_set,
    opp_welfare_set,
    own_player_idx,
    meta_payoff,
    payoff,
    payoff_default,
):
    if len(own_welfare_set & opp_welfare_set) > 0:
        assert meta_payoff[own_player_idx] == payoff
    else:
        assert meta_payoff[own_player_idx] == payoff_default


def _add_nominal_case(welfares, all_welfare_pairs_wt_payoffs, own_player_idx):
    own_welfare_set = set(random.sample(welfares, 1))
    opp_welfare_set = set(random.sample(welfares, 1))
    welfare_pair_name = MetaGameSolver.from_pair_of_welfare_names_to_key(
        list(own_welfare_set)[0], list(opp_welfare_set)[0]
    )
    payoff = random.random()
    all_welfare_pairs_wt_payoffs[welfare_pair_name] = [-1, -1]
    all_welfare_pairs_wt_payoffs[welfare_pair_name][own_player_idx] = payoff

    return own_welfare_set, opp_welfare_set, payoff


def _add_default_case(welfares, all_welfare_pairs_wt_payoffs, own_player_idx):
    own_default_welfare_fn = random.sample(welfares, 1)[0]
    opp_default_welfare_fn = random.sample(welfares, 1)[0]
    welfare_default_pair_name = (
        MetaGameSolver.from_pair_of_welfare_names_to_key(
            own_default_welfare_fn, opp_default_welfare_fn
        )
    )
    payoff_default = random.random()

    all_welfare_pairs_wt_payoffs[welfare_default_pair_name] = [-1, -1]
    all_welfare_pairs_wt_payoffs[welfare_default_pair_name][
        own_player_idx
    ] = payoff_default

    return own_default_welfare_fn, opp_default_welfare_fn, payoff_default


def test__list_all_set_of_welfare_fn():
    for _ in range(100):
        (
            own_player_idx,
            opp_player_idx,
            welfares,
            all_welfare_pairs_wt_payoffs,
        ) = _given_n_welfare_fn()
        meta_game_solver = _when_setting_the_game(
            all_welfare_pairs_wt_payoffs, own_player_idx, opp_player_idx
        )
        _assert_right_number_of_sets_and_presence_of_single_and_pairs(
            meta_game_solver, welfares
        )


def _given_n_welfare_fn():
    n_welfares = _get_random_number_of_welfare_fn()
    welfares = ["welfare_" + str(el) for el in list(range(n_welfares))]
    own_player_idx = _get_random_position_of_players()
    opp_player_idx = (own_player_idx + 1) % 2

    all_welfare_pairs_wt_payoffs = (
        _get_all_welfare_pairs_wt_extrem_payoffs_for_i(
            welfares=welfares,
            own_player_idx=own_player_idx,
            best_welfare=welfares[0],
        )
    )
    return (
        own_player_idx,
        opp_player_idx,
        welfares,
        all_welfare_pairs_wt_payoffs,
    )


def _when_setting_the_game(
    all_welfare_pairs_wt_payoffs, own_player_idx, opp_player_idx
):
    meta_game_solver = MetaGameSolver()
    meta_game_solver.setup_meta_game(
        all_welfare_pairs_wt_payoffs,
        own_player_idx=own_player_idx,
        opp_player_idx=opp_player_idx,
        own_default_welfare_fn="welfare_0",
        opp_default_welfare_fn="welfare_1",
    )
    return meta_game_solver


def _assert_right_number_of_sets_and_presence_of_single_and_pairs(
    meta_game_solver, welfares
):
    meta_game_solver._list_all_set_of_welfare_fn()
    if len(welfares) == 2:
        assert len(meta_game_solver.welfare_fn_sets) == 3
    elif len(welfares) == 3:
        assert len(meta_game_solver.welfare_fn_sets) == 3 + 3 + 1
    elif len(welfares) == 4:
        print(
            "meta_game_solver.welfare_fn_sets",
            meta_game_solver.welfare_fn_sets,
        )
        assert len(meta_game_solver.welfare_fn_sets) == 4 + 6 + 4 + 1
    for welfare in welfares:
        assert frozenset([welfare]) in meta_game_solver.welfare_fn_sets
        for welfare_2 in welfares:
            assert (
                frozenset([welfare, welfare_2])
                in meta_game_solver.welfare_fn_sets
            )
