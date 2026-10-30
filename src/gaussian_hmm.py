"""Hidden Markov model with Gaussian emissions, written with numpy only.

Every emission has a diagonal covariance: for each hidden state and each feature there is
one average and one variance, and the features are treated as independent given the state.
All probabilities are stored and computed as natural logarithms, because the product of
many probabilities underflows to zero in floating point.

Shapes used throughout: T windows, N hidden states, D features.
"""

import numpy as np


def logsumexp(values, axis=None):
    """Return log(sum(exp(values))) along an axis without overflow or underflow."""
    values = np.asarray(values, dtype=float)
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):  # log(0) gives -inf, which is the correct answer here
        total = np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True)) + peak
    return np.squeeze(total) if axis is None else np.squeeze(total, axis=axis)


class GaussianHMM:
    """Hidden Markov model with N hidden states and D-dimensional Gaussian emissions."""

    def __init__(self, n_states, n_features, variance_floor=1e-3):
        """Create a model with uniform probabilities, zero averages and unit variances.

        `variance_floor` is the smallest variance allowed, which keeps a state from becoming
        infinitely sharp on a feature that barely varies.
        """
        self.n_states = n_states
        self.n_features = n_features
        self.variance_floor = variance_floor
        self.log_start = np.full(n_states, -np.log(n_states))
        self.log_trans = np.full((n_states, n_states), -np.log(n_states))
        self.means = np.zeros((n_states, n_features))
        self.variances = np.ones((n_states, n_features))

    def set_parameters(self, start, trans, means, variances):
        """Set the model from ordinary (not logarithmic) probabilities and Gaussian parameters.

        `start` has shape (N,), `trans` (N, N) with rows that sum to 1, `means` and
        `variances` (N, D). Variances below the variance floor are raised to it.
        """
        start, trans = np.asarray(start, float), np.asarray(trans, float)
        means, variances = np.asarray(means, float), np.asarray(variances, float)
        n, d = self.n_states, self.n_features
        if start.shape != (n,) or trans.shape != (n, n):
            raise ValueError("start must have shape (N,) and trans shape (N, N)")
        if means.shape != (n, d) or variances.shape != (n, d):
            raise ValueError("means and variances must have shape (N, D)")
        if not np.isclose(start.sum(), 1.0) or not np.allclose(trans.sum(axis=1), 1.0):
            raise ValueError("start and every row of trans must sum to 1")
        if (start < 0).any() or (trans < 0).any():
            raise ValueError("probabilities cannot be negative")
        with np.errstate(divide="ignore"):
            self.log_start = np.log(start)
            self.log_trans = np.log(trans)
        self.means = means.copy()
        self.variances = np.maximum(variances, self.variance_floor)

    def log_emission(self, observations):
        """Log density of every window under every state, shape (T, N)."""
        x = self._check(observations)
        difference = x[:, None, :] - self.means[None, :, :]
        log_density = -0.5 * (np.log(2 * np.pi * self.variances)[None, :, :]
                              + difference ** 2 / self.variances[None, :, :])
        return log_density.sum(axis=2)

    def forward(self, observations):
        """Forward algorithm in log-space.

        Returns `log_alpha` of shape (T, N), where `log_alpha[t, j]` is the log probability of
        the first t+1 windows together with being in state j at window t, and the log
        likelihood of the whole sequence, which is the sum over all hidden paths.
        """
        log_b = self.log_emission(observations)
        log_alpha = np.empty_like(log_b)
        log_alpha[0] = self.log_start + log_b[0]
        for t in range(1, len(log_b)):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + self.log_trans, axis=0) + log_b[t]
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def viterbi(self, observations):
        """Viterbi decoding in log-space: the single most likely sequence of hidden states.

        Returns the path as an integer array of length T and the log probability of that path
        together with the observations. For every window and state, the best of the incoming
        routes is kept and its origin is stored as a backpointer; the path is read out by
        following the backpointers from the best final state. Ties go to the lowest state number.
        """
        log_b = self.log_emission(observations)
        n_windows = len(log_b)
        score = np.empty_like(log_b)
        backpointer = np.zeros(log_b.shape, dtype=int)
        score[0] = self.log_start + log_b[0]
        for t in range(1, n_windows):
            routes = score[t - 1][:, None] + self.log_trans  # routes[i, j]: from state i into state j
            backpointer[t] = routes.argmax(axis=0)
            score[t] = routes.max(axis=0) + log_b[t]
        path = np.empty(n_windows, dtype=int)
        path[-1] = score[-1].argmax()
        for t in range(n_windows - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        return path, float(score[-1].max())

    def backward(self, observations):
        """Backward algorithm in log-space.

        `log_beta[t, i]` is the log probability of the windows after t, given that the model is
        in state i at window t. The last row is zero (log of 1), since nothing follows it.
        """
        log_b = self.log_emission(observations)
        log_beta = np.zeros_like(log_b)
        for t in range(len(log_b) - 2, -1, -1):
            log_beta[t] = logsumexp(self.log_trans + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def posteriors(self, observations):
        """The E-step for one sequence.

        Returns `gamma` of shape (T, N), the probability of each state at each window given the
        whole sequence; the expected number of transitions between every pair of states,
        shape (N, N); and the log likelihood of the sequence.
        """
        x = self._check(observations)
        log_b = self.log_emission(x)
        log_alpha, log_likelihood = self.forward(x)
        log_beta = self.backward(x)
        gamma = np.exp(log_alpha + log_beta - log_likelihood)
        expected_transitions = np.zeros((self.n_states, self.n_states))
        for t in range(len(x) - 1):
            log_xi = (log_alpha[t][:, None] + self.log_trans
                      + (log_b[t + 1] + log_beta[t + 1])[None, :] - log_likelihood)
            expected_transitions += np.exp(log_xi)
        return gamma, expected_transitions, log_likelihood

    def fit(self, sequences, max_iterations=200, tolerance=1e-4, trans_prior=None, start_prior=None):
        """Baum-Welch training on a list of sequences, each an array of shape (T_k, D).

        Every iteration runs the E-step (state probabilities from the current parameters) and the
        M-step (parameters re-estimated as weighted averages). `trans_prior` (N, N) and
        `start_prior` (N,) are pretend counts added to the expected transition and start counts,
        which keeps switching possible even when the data never switches.

        Training stops when the objective, the log likelihood plus the log of the prior, rises by
        less than `tolerance`, or after `max_iterations`. In exact arithmetic this objective never
        decreases, so a decrease is recorded in the result as a sign of a problem (the variance
        floor can cause a very small one).

        Returns a dictionary with `history` (one entry per iteration with the log likelihood, the
        objective and its change), `converged`, and `objective_never_decreased`.
        """
        sequences = [self._check(s) for s in sequences]
        n = self.n_states
        trans_prior = np.zeros((n, n)) if trans_prior is None else np.asarray(trans_prior, float)
        start_prior = np.zeros(n) if start_prior is None else np.asarray(start_prior, float)
        if trans_prior.shape != (n, n) or start_prior.shape != (n,):
            raise ValueError("trans_prior must have shape (N, N) and start_prior shape (N,)")
        if (trans_prior < 0).any() or (start_prior < 0).any():
            raise ValueError("prior counts cannot be negative")

        history, converged, previous = [], False, None
        for iteration in range(1, max_iterations + 1):
            results = [self.posteriors(s) for s in sequences]
            log_likelihood = sum(r[2] for r in results)
            objective = log_likelihood + self._log_prior(trans_prior, start_prior)
            change = None if previous is None else objective - previous
            history.append({"iteration": iteration, "log_likelihood": log_likelihood,
                            "objective": objective, "change": change})
            if change is not None and change < tolerance:
                converged = True
                break
            self._maximization(sequences, results, trans_prior, start_prior)
            previous = objective
        decreases = [h["change"] for h in history if h["change"] is not None and h["change"] < -1e-9]
        return {"history": history, "converged": converged, "objective_never_decreased": not decreases}

    def _log_prior(self, trans_prior, start_prior):
        """Log of the prior pretend counts, up to a constant: sum of count * log probability."""
        trans_used, start_used = trans_prior > 0, start_prior > 0  # skip 0 * log(0) terms
        trans_term = (trans_prior[trans_used] * self.log_trans[trans_used]).sum()
        start_term = (start_prior[start_used] * self.log_start[start_used]).sum()
        return float(trans_term + start_term)

    def _maximization(self, sequences, results, trans_prior, start_prior):
        """The M-step: new parameters from the state probabilities of the E-step."""
        start_counts = sum(r[0][0] for r in results) + start_prior
        trans_counts = sum(r[1] for r in results) + trans_prior
        start = start_counts / start_counts.sum()
        row_totals = trans_counts.sum(axis=1, keepdims=True)
        old_trans = np.exp(self.log_trans)
        trans = np.where(row_totals > 0, trans_counts / np.where(row_totals > 0, row_totals, 1.0), old_trans)

        weights = sum(r[0].sum(axis=0) for r in results)  # total probability mass of each state
        means, variances = self.means.copy(), self.variances.copy()
        for state in range(self.n_states):
            if weights[state] < 1e-10:  # a state that explains nothing keeps its old parameters
                continue
            weighted_sum = sum(r[0][:, state] @ s for r, s in zip(results, sequences))
            means[state] = weighted_sum / weights[state]
            squares = sum(r[0][:, state] @ ((s - means[state]) ** 2) for r, s in zip(results, sequences))
            variances[state] = squares / weights[state]
        self.set_parameters(start, trans, means, variances)

    def _check(self, observations):
        """Return the observations as a float array of shape (T, D), or raise a ValueError."""
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"observations must have shape (T, {self.n_features})")
        if x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError("observations must be non-empty and finite")
        return x
