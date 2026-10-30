"""Checks for gaussian_hmm.py. Run from the repository folder with: python tests/test_gaussian_hmm.py"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gaussian_hmm import GaussianHMM, logsumexp  # noqa: E402


def random_model(n_states, n_features, seed):
    """A model with random but valid parameters."""
    rng = np.random.default_rng(seed)
    model = GaussianHMM(n_states, n_features, variance_floor=1e-3)
    model.set_parameters(
        start=rng.dirichlet(np.ones(n_states)),
        trans=rng.dirichlet(np.ones(n_states), size=n_states),
        means=rng.normal(size=(n_states, n_features)),
        variances=rng.uniform(0.2, 2.0, size=(n_states, n_features)),
    )
    return model


def brute_force_log_likelihood(model, x):
    """Add the probabilities of all N**T hidden paths, one by one."""
    log_b = model.log_emission(x)
    total = 0.0
    for path in itertools.product(range(model.n_states), repeat=len(x)):
        log_p = model.log_start[path[0]] + log_b[0, path[0]]
        for t in range(1, len(x)):
            log_p += model.log_trans[path[t - 1], path[t]] + log_b[t, path[t]]
        total += math.exp(log_p)
    return math.log(total)


def brute_force_best_path(model, x):
    """Score all N**T hidden paths one by one and return the best path and its log probability."""
    log_b = model.log_emission(x)
    best_path, best_log_p = None, -np.inf
    for path in itertools.product(range(model.n_states), repeat=len(x)):
        log_p = model.log_start[path[0]] + log_b[0, path[0]]
        for t in range(1, len(x)):
            log_p += model.log_trans[path[t - 1], path[t]] + log_b[t, path[t]]
        if log_p > best_log_p:
            best_path, best_log_p = path, log_p
    return np.array(best_path), best_log_p


def test_logsumexp():
    assert np.isclose(logsumexp([math.log(0.2), math.log(0.3)]), math.log(0.5))
    assert np.isclose(logsumexp([-1000.0, -1000.0]), -1000.0 + math.log(2))
    assert logsumexp([-np.inf, -np.inf]) == -np.inf


def test_log_emission_matches_formula():
    model = GaussianHMM(2, 2)
    model.set_parameters([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [[0.0, 1.0], [2.0, -1.0]], [[1.0, 2.0], [0.5, 4.0]])
    x = np.array([[0.3, 0.7]])
    expected = [
        sum(-0.5 * (math.log(2 * math.pi * v) + (xi - m) ** 2 / v) for xi, m, v in zip(x[0], model.means[s], model.variances[s]))
        for s in range(2)
    ]
    assert np.allclose(model.log_emission(x)[0], expected)


def test_forward_equals_brute_force():
    for seed in range(5):
        model = random_model(n_states=4, n_features=3, seed=seed)
        x = np.random.default_rng(100 + seed).normal(size=(5, 3))
        _, log_likelihood = model.forward(x)
        assert np.isclose(log_likelihood, brute_force_log_likelihood(model, x), atol=1e-10)


def test_viterbi_equals_brute_force():
    for seed in range(10):
        model = random_model(n_states=4, n_features=3, seed=seed)
        x = np.random.default_rng(200 + seed).normal(size=(6, 3))
        path, log_p = model.viterbi(x)
        best_path, best_log_p = brute_force_best_path(model, x)
        assert np.array_equal(path, best_path)
        assert np.isclose(log_p, best_log_p, atol=1e-10)


def test_best_path_is_never_better_than_all_paths_together():
    for seed in range(5):
        model = random_model(n_states=3, n_features=2, seed=seed)
        x = np.random.default_rng(300 + seed).normal(size=(8, 2))
        assert model.viterbi(x)[1] <= model.forward(x)[1] + 1e-12


def test_viterbi_respects_forbidden_transitions_and_start():
    model = GaussianHMM(3, 1)
    trans = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])  # only 0->1->2->0 or stay
    model.set_parameters([1.0, 0.0, 0.0], trans, [[0.0], [5.0], [10.0]], [[1.0], [1.0], [1.0]])
    x = np.array([[10.0], [10.0], [0.0], [10.0]])  # the data tries to start in state 2 and jump 0 -> 2
    path, log_p = model.viterbi(x)
    assert path[0] == 0 and np.isfinite(log_p)
    for before, after in zip(path, path[1:]):
        assert trans[before, after] > 0


def test_viterbi_output_shape_and_range():
    model = random_model(4, 5, seed=3)
    path, _ = model.viterbi(np.random.default_rng(4).normal(size=(50, 5)))
    assert path.shape == (50,) and path.min() >= 0 and path.max() < 4


def sample_sequence(model, length, rng):
    """Draw a hidden path and its observations from a model."""
    start, trans = np.exp(model.log_start), np.exp(model.log_trans)
    state = rng.choice(model.n_states, p=start)
    states, observations = [], []
    for _ in range(length):
        states.append(state)
        observations.append(rng.normal(model.means[state], np.sqrt(model.variances[state])))
        state = rng.choice(model.n_states, p=trans[state])
    return np.array(states), np.array(observations)


def test_forward_backward_identity():
    model = random_model(4, 3, seed=11)
    x = np.random.default_rng(12).normal(size=(9, 3))
    log_alpha, log_likelihood = model.forward(x)
    log_beta = model.backward(x)
    for t in range(len(x)):
        assert np.isclose(np.logaddexp.reduce(log_alpha[t] + log_beta[t]), log_likelihood, atol=1e-10)


def test_posteriors_are_consistent():
    model = random_model(4, 3, seed=13)
    x = np.random.default_rng(14).normal(size=(12, 3))
    gamma, expected_transitions, _ = model.posteriors(x)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.isclose(expected_transitions.sum(), len(x) - 1)
    # the expected transitions out of the states at windows 0..T-2 add up to their probabilities
    assert np.allclose(expected_transitions.sum(axis=1), gamma[:-1].sum(axis=0))


def test_posteriors_equal_brute_force_over_all_paths():
    model = random_model(3, 2, seed=31)
    x = np.random.default_rng(32).normal(size=(5, 2))
    log_b = model.log_emission(x)
    weights, paths = [], list(itertools.product(range(3), repeat=len(x)))
    for path in paths:
        log_p = model.log_start[path[0]] + log_b[0, path[0]]
        for t in range(1, len(x)):
            log_p += model.log_trans[path[t - 1], path[t]] + log_b[t, path[t]]
        weights.append(math.exp(log_p))
    weights = np.array(weights) / sum(weights)
    gamma_expected = np.zeros((len(x), 3))
    transitions_expected = np.zeros((3, 3))
    for weight, path in zip(weights, paths):
        for t, state in enumerate(path):
            gamma_expected[t, state] += weight
        for before, after in zip(path, path[1:]):
            transitions_expected[before, after] += weight
    gamma, expected_transitions, _ = model.posteriors(x)
    assert np.allclose(gamma, gamma_expected, atol=1e-10)
    assert np.allclose(expected_transitions, transitions_expected, atol=1e-10)


def test_one_state_m_step_is_ordinary_mean_and_variance():
    rng = np.random.default_rng(15)
    x = rng.normal(loc=3.0, scale=2.0, size=(200, 2))
    model = GaussianHMM(1, 2, variance_floor=1e-6)
    model.fit([x], tolerance=1e-12)
    assert np.allclose(model.means[0], x.mean(axis=0))
    assert np.allclose(model.variances[0], x.var(axis=0))


def test_baum_welch_never_decreases_the_objective():
    rng = np.random.default_rng(16)
    truth = random_model(3, 2, seed=17)
    sequences = [sample_sequence(truth, 60, rng)[1] for _ in range(5)]
    for prior in (None, np.full((3, 3), 2.0)):
        model = random_model(3, 2, seed=18)
        result = model.fit(sequences, max_iterations=60, tolerance=0.0, trans_prior=prior)
        objective = [h["objective"] for h in result["history"]]
        assert result["objective_never_decreased"]
        assert all(b >= a - 1e-9 for a, b in zip(objective, objective[1:]))


def test_baum_welch_recovers_known_parameters():
    rng = np.random.default_rng(19)
    truth = GaussianHMM(2, 2)
    truth.set_parameters([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], [[-3.0, -3.0], [3.0, 3.0]], [[1.0, 1.0], [1.0, 1.0]])
    sequences = [sample_sequence(truth, 100, rng)[1] for _ in range(20)]
    model = GaussianHMM(2, 2)
    model.set_parameters([0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], [[-1.0, -1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]])
    result = model.fit(sequences, max_iterations=200, tolerance=1e-6)
    assert result["converged"]
    assert np.allclose(model.means, truth.means, atol=0.3)
    assert np.allclose(model.variances, truth.variances, atol=0.4)
    assert np.allclose(np.exp(model.log_trans), np.exp(truth.log_trans), atol=0.06)


def test_convergence_stops_before_the_iteration_limit():
    rng = np.random.default_rng(20)
    x = np.vstack([rng.normal(-2, 0.5, size=(40, 1)), rng.normal(2, 0.5, size=(40, 1))])
    model = GaussianHMM(2, 1)
    model.set_parameters([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [[-1.0], [1.0]], [[1.0], [1.0]])
    result = model.fit([x], max_iterations=500, tolerance=1e-6)
    assert result["converged"] and len(result["history"]) < 500
    assert result["history"][-1]["change"] < 1e-6


def test_prior_keeps_switching_possible_when_the_data_never_switches():
    rng = np.random.default_rng(21)
    sequences = [rng.normal(-3, 0.5, size=(30, 1)), rng.normal(3, 0.5, size=(30, 1))]  # one state each, no switch
    prior = np.full((2, 2), 0.2) + np.eye(2) * 1.6  # 90 percent stay, 10 percent switch, 2 pretend counts per row
    with_prior = GaussianHMM(2, 1)
    with_prior.set_parameters([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [[-1.0], [1.0]], [[1.0], [1.0]])
    with_prior.fit(sequences, trans_prior=prior, max_iterations=100)
    without = GaussianHMM(2, 1)
    without.set_parameters([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [[-1.0], [1.0]], [[1.0], [1.0]])
    without.fit(sequences, max_iterations=100)
    assert np.exp(with_prior.log_trans)[0, 1] > 0.005
    assert np.exp(without.log_trans)[0, 1] < 1e-3


def test_training_with_a_zero_probability_emits_no_warning():
    """A transition that is exactly zero must not make the prior term produce NaN warnings."""
    import warnings
    model = GaussianHMM(3, 1)
    trans = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    model.set_parameters([1.0, 0.0, 0.0], trans, [[0.0], [5.0], [10.0]], [[1.0], [1.0], [1.0]])
    x = np.array([[0.1], [4.9], [10.2], [0.3], [5.1]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = model.fit([x], max_iterations=5)
        model.fit([x], max_iterations=5, trans_prior=np.array([[1.0, 1.0, 0.0]] * 3), start_prior=np.array([0.0, 1.0, 1.0]))
    assert np.isfinite(result["history"][0]["objective"])


def test_forward_is_stable_on_long_sequences():
    model = random_model(n_states=4, n_features=12, seed=7)
    x = np.random.default_rng(8).normal(size=(5000, 12)) * 3
    _, log_likelihood = model.forward(x)
    assert np.isfinite(log_likelihood) and log_likelihood < -1000


def test_invalid_input_is_rejected():
    model = random_model(3, 2, seed=1)
    for bad in (np.zeros((4, 3)), np.array([[np.nan, 0.0]]), np.zeros((0, 2))):
        try:
            model.forward(bad)
        except ValueError:
            continue
        raise AssertionError("invalid observations were accepted")
    try:
        model.set_parameters([0.5, 0.5, 0.5], np.full((3, 3), 1 / 3), np.zeros((3, 2)), np.ones((3, 2)))
    except ValueError:
        return
    raise AssertionError("start probabilities that do not sum to 1 were accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"{len(tests)} tests passed")
