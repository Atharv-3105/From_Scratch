import numpy as np
import pytest

from sim.env import WarmPoolEnv, parse_actions, pool_for_action, pool_cap, reward_scale
from utils.config import load_config

CONFIG = {
    "reference_window": 5,
    "rho": 10,
    "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}
ACTIONS = parse_actions(CONFIG["actions"])


def run(env, actions):
    """Play a list of actions, return the list of per-step info dicts."""
    infos = []
    for action in actions:
        _, _, _, info = env.step(action)
        infos.append(info)
    return infos


# ---- action -> pool size ----
@pytest.mark.parametrize("ref, expected", [
    (0, [0, 1, 0, 0, 0, 0]),     # nothing recently: only "keep one warm" gives a pool
    (1, [0, 1, 1, 1, 2, 2]),     # six actions, only three different pools
    (3, [0, 1, 2, 3, 5, 6]),     # ceil(1.5) = 2 and ceil(4.5) = 5
    (4, [0, 1, 2, 4, 6, 8]),
])
def test_action_mapping(ref, expected):
    assert [pool_for_action(a, ref, 10_000, ACTIONS) for a in range(6)] == expected


def test_pool_is_capped():
    assert [pool_for_action(a, 3, 4, ACTIONS) for a in range(6)] == [0, 1, 2, 3, 4, 4]


def test_shipped_config_has_the_six_actions():
    config = load_config("configs/sim.yaml")
    assert [pool_for_action(a, 3, 10_000, parse_actions(config["actions"])) for a in range(6)] == [0, 1, 2, 3, 5, 6]


# ---- a trace computed by hand ----
def test_hand_computed_trace():
    # rho=10, kappa=0.1, scale=2, 1024 MB (= 1 GB). Always action 3 (pool = R).
    env = WarmPoolEnv(CONFIG)
    env.reset([0, 3, 3, 0, 1], scale=2, mem_mb=1024)
    infos = run(env, [3] * 5)

    assert [i["ref_demand"] for i in infos] == [0, 0, 3, 3, 3]
    assert [i["pool"] for i in infos] == [0, 0, 3, 3, 3]
    assert [i["cold"] for i in infos] == [0, 3, 0, 0, 0]
    assert [i["idle"] for i in infos] == [0, 0, 0, 3, 2]
    assert [i["start"] for i in infos] == [0, 0, 3, 0, 0]
    expected_rewards = [0.0, -15.0, -0.15, -1.5, -1.0]    # -(10*cold + idle + 0.1*start) / 2
    assert [i["reward"] for i in infos] == pytest.approx(expected_rewards)

    total = env.summary()
    assert total["cold"] == 3 and total["idle"] == 5 and total["start"] == 3
    assert total["idle_gbmin"] == pytest.approx(5.0)
    assert total["reward"] == pytest.approx(-17.65)


def test_zero_kappa_makes_starting_free():
    env = WarmPoolEnv({**CONFIG, "kappa": 0})
    env.reset([0, 3, 3, 0, 1], scale=2)
    run(env, [3] * 5)
    assert env.summary()["reward"] == pytest.approx(-17.5)


# ---- the traps ----
def test_reference_window_is_exactly_five_minutes():
    env = WarmPoolEnv(CONFIG)
    env.reset([4, 0, 0, 0, 0, 0, 0])
    infos = run(env, [3] * 7)
    # minute 5 still sees minute 0 (window = minutes 0..4); minute 6 no longer does
    assert [i["ref_demand"] for i in infos] == [0, 4, 4, 4, 4, 4, 0]


def test_no_lookahead():
    # Changing demand at minute 10 or later must not change anything decided before minute 10 or AT minute 10.
    base = np.zeros(20, dtype=int)
    base[3] = 2
    changed = base.copy()
    changed[10:] = 50
    first, second = WarmPoolEnv(CONFIG), WarmPoolEnv(CONFIG)
    first.reset(base)
    second.reset(changed)
    a = run(first, [4] * 11)
    b = run(second, [4] * 11)
    assert [(i["ref_demand"], i["pool"]) for i in a] == [(i["ref_demand"], i["pool"]) for i in b]


def test_previous_day_history_feeds_the_first_minutes():
    env = WarmPoolEnv(CONFIG)
    state = env.reset([0, 0, 0], previous_demand=[1, 2, 7, 1, 1, 1])   # only the last 5 count
    assert state["ref_demand"] == 7
    env.reset([0, 0, 0], previous_demand=[9, 1, 1, 1, 1, 1])           # the 9 is 6 minutes back: forgotten
    assert env.step(3)[3]["ref_demand"] == 1


def test_initial_pool_changes_the_first_start_cost():
    env = WarmPoolEnv(CONFIG)
    env.reset([2, 2], previous_demand=[2], initial_pool=2)
    info = env.step(3)[3]                 # R = 2 -> pool 2, already running: nothing to start
    assert info["start"] == 0


def test_scaling_down_costs_nothing_in_starts():
    env = WarmPoolEnv(CONFIG)
    env.reset([5, 0, 0, 0, 0, 0, 0, 0], previous_demand=[5, 5])
    infos = run(env, [3] * 8)
    assert sum(i["start"] for i in infos) == 5          # one start-up at minute 0, never charged again


# ---- the episode itself ----
def test_episode_ends_and_rejects_more_steps():
    env = WarmPoolEnv(CONFIG)
    env.reset([1, 1])
    assert env.step(3)[2] is False
    assert env.step(3)[2] is True
    with pytest.raises(RuntimeError):
        env.step(3)


def test_bad_action_is_rejected():
    env = WarmPoolEnv(CONFIG)
    env.reset([1])
    with pytest.raises(ValueError):
        env.step(6)


def test_pool_never_exceeds_the_cap():
    env = WarmPoolEnv(CONFIG)
    env.reset([100, 100, 100], max_pool=7)
    infos = run(env, [5, 5, 5])
    assert max(i["pool"] for i in infos) == 7


# ---- per-function constants ----
def test_pool_cap_and_reward_scale():
    assert pool_cap([0, 0, 0]) == 1                 # at least 1
    assert pool_cap([0, 4, 2]) == 8                 # twice the biggest training demand
    assert reward_scale([0, 0, 0]) == 1.0           # nothing positive -> 1
    assert reward_scale([0, 2, 4]) == 3.0           # mean of the POSITIVE values
    assert reward_scale([0, 1, 1]) == 1.0           # never below 1