import numpy as np
import pytest

from sim.env import WarmPoolEnv
from sim.vec_env import VecWarmPoolEnv

CONFIG = {
    "reference_window": 5,
    "rho": 10,
    "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}
FIELDS = ["minute", "ref_demand", "pool", "demand", "cold", "idle", "start", "idle_gbmin", "reward"]


def play(env, actions):
    """actions: (steps, episodes). Returns {field: array (steps, episodes)} of the per-step info."""
    steps = []
    for row in actions:
        _, _, _, info = env.step(row)
        steps.append(info)
    return {name: np.array([s[name] for s in steps]) for name in FIELDS}


def play_scalar(env, actions):
    """Same as play, for the single-function env: actions is a list of ints."""
    steps = [env.step(int(action))[3] for action in actions]
    return {name: np.array([s[name] for s in steps]) for name in FIELDS}


# ---- the main guarantee: same numbers as the single-function environment ----
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_scalar_env_exactly(seed):
    rng = np.random.default_rng(seed)
    episodes, minutes = 40, 80
    demand = rng.integers(0, 6, size=(episodes, minutes))
    demand[rng.random((episodes, minutes)) < 0.1] = rng.integers(10, 40)       # some spikes
    demand[rng.random((episodes, minutes)) < 0.4] = 0                          # many quiet minutes
    batch = {
        "demand": demand,
        "scale": rng.uniform(1, 5, size=episodes),
        "max_pool": rng.integers(1, 40, size=episodes),
        "mem_mb": rng.uniform(50, 500, size=episodes),
        "previous_demand": rng.integers(0, 8, size=(episodes, 5)),
        "initial_pool": rng.integers(0, 5, size=episodes),
    }
    actions = rng.integers(0, 6, size=(minutes, episodes))

    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(**batch)
    got = play(vec, actions)

    for i in range(episodes):
        scalar = WarmPoolEnv(CONFIG)
        scalar.reset(demand[i], scale=batch["scale"][i], max_pool=int(batch["max_pool"][i]),
                     mem_mb=batch["mem_mb"][i], previous_demand=batch["previous_demand"][i],
                     initial_pool=int(batch["initial_pool"][i]))
        want = play_scalar(scalar, actions[:, i])
        for name in FIELDS:
            mine = got[name] if name == "minute" else got[name][:, i]    # "minute" is one number per step
            np.testing.assert_allclose(mine, want[name], rtol=1e-12, err_msg=name)
        totals = scalar.summary()
        for name, value in totals.items():
            np.testing.assert_allclose(vec.summary()[name][i], value, rtol=1e-12, err_msg=name)


# ---- hand-computed, several episodes at once ----
def test_hand_computed_trace_in_a_batch():
    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(demand=[[0, 3, 3, 0, 1], [2, 2, 2, 2, 2]], scale=[2, 1], mem_mb=[1024, 0])
    got = play(vec, np.array([[3, 3]] * 5))
    assert got["pool"][:, 0].tolist() == [0, 0, 3, 3, 3]
    assert got["cold"][:, 0].tolist() == [0, 3, 0, 0, 0]
    assert got["idle"][:, 0].tolist() == [0, 0, 0, 3, 2]
    assert got["reward"][:, 0] == pytest.approx([0.0, -15.0, -0.15, -1.5, -1.0])
    assert vec.summary()["reward"][0] == pytest.approx(-17.65)
    # episode 2: R=0 at minute 0, then 2 forever -> cold 2 at minute 0, pool 2 from minute 1 on
    assert got["pool"][:, 1].tolist() == [0, 2, 2, 2, 2]
    assert got["cold"][:, 1].tolist() == [2, 0, 0, 0, 0]


def test_each_episode_uses_its_own_action():
    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(demand=np.zeros((6, 2), dtype=int), previous_demand=np.full((6, 5), 3))
    _, _, _, info = vec.step(np.arange(6))                 # episode k plays action k
    assert info["pool"].tolist() == [0, 1, 2, 3, 5, 6]     # R = 3 for everyone


def test_pool_is_capped_per_episode():
    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(demand=np.zeros((2, 1), dtype=int), previous_demand=np.full((2, 5), 10), max_pool=[4, 100])
    _, _, _, info = vec.step(np.array([5, 5]))
    assert info["pool"].tolist() == [4, 20]


# ---- the traps ----
def test_reference_window_is_exactly_five_minutes():
    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(demand=[[4, 0, 0, 0, 0, 0, 0]])
    refs = [vec.step(np.array([3]))[3]["ref_demand"][0] for _ in range(7)]
    assert refs == [0, 4, 4, 4, 4, 4, 0]


def test_no_lookahead():
    base = np.zeros((1, 20), dtype=int)
    base[0, 3] = 2
    changed = base.copy()
    changed[0, 10:] = 50
    a, b = VecWarmPoolEnv(CONFIG), VecWarmPoolEnv(CONFIG)
    a.reset(base)
    b.reset(changed)
    first = play(a, np.full((11, 1), 4))
    second = play(b, np.full((11, 1), 4))
    np.testing.assert_array_equal(first["ref_demand"], second["ref_demand"])
    np.testing.assert_array_equal(first["pool"], second["pool"])


def test_same_inputs_give_same_outputs():
    rng = np.random.default_rng(5)
    demand = rng.integers(0, 9, size=(30, 50))
    actions = rng.integers(0, 6, size=(50, 30))
    results = []
    for _ in range(2):
        env = VecWarmPoolEnv(CONFIG)
        env.reset(demand)
        results.append(play(env, actions))
    for name in FIELDS:
        np.testing.assert_array_equal(results[0][name], results[1][name])


# ---- the episode itself ----
def test_done_flag_and_extra_steps():
    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(demand=[[1, 1]])
    assert vec.step(np.array([3]))[2] is False
    assert vec.step(np.array([3]))[2] is True
    with pytest.raises(RuntimeError):
        vec.step(np.array([3]))


def test_bad_actions_are_rejected():
    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(demand=[[1], [1]])
    with pytest.raises(ValueError):
        vec.step(np.array([3]))             # one action for two episodes
    with pytest.raises(ValueError):
        vec.step(np.array([3, 6]))          # action 6 does not exist
        
# ---- setting the pool directly (baselines, the oracle) ----
def test_step_pool_gives_the_same_result_as_the_action_that_produced_that_pool():
    rng = np.random.default_rng(7)
    demand = rng.integers(0, 6, size=(20, 60))
    actions = rng.integers(0, 6, size=(60, 20))
    by_action = VecWarmPoolEnv(CONFIG)
    by_action.reset(demand, max_pool=9)
    chosen = play(by_action, actions)
    by_pool = VecWarmPoolEnv(CONFIG)
    by_pool.reset(demand, max_pool=9)
    for t in range(60):
        by_pool.step_pool(chosen["pool"][t])
    for name, values in by_action.summary().items():
        np.testing.assert_array_equal(values, by_pool.summary()[name])


def test_step_pool_applies_the_cap_and_rejects_bad_input():
    vec = VecWarmPoolEnv(CONFIG)
    vec.reset(demand=np.zeros((2, 3), dtype=int), max_pool=[4, 100])
    assert vec.step_pool(np.array([50, 50]))[3]["pool"].tolist() == [4, 50]
    with pytest.raises(ValueError):
        vec.step_pool(np.array([-1, 0]))
    with pytest.raises(TypeError):
        vec.step_pool(np.array([1.5, 2.0]))
    with pytest.raises(ValueError):
        vec.step_pool(np.array([1]))