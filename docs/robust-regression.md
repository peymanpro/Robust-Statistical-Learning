# Robust Regression

Ordinary least squares assigns quadratic loss to every residual. A single
large residual can therefore dominate the fitted parameters.

This project adds Huber regression through iteratively reweighted least squares
(IRLS). The Huber loss is quadratic near zero and linear in the tails:

    rho_delta(r) =
        0.5 r^2                         if |r| <= delta
        delta (|r| - 0.5 delta)         otherwise.

The corresponding IRLS weights are:

    w_i = 1                             if |r_i| <= delta
          delta / |r_i|                 otherwise.

Each weighted subproblem is solved by the existing SVD least-squares solver,
so the robust layer inherits the numerical safeguards of the project.

## Why this matters here

Robustness is treated as a statistical property as well as a numerical one:

- numerical stability asks whether finite precision changes a mathematically
  equivalent computation;
- statistical robustness asks whether small amounts of contaminated data can
  change the fitted model disproportionately.

These are different failure modes and are measured separately.

## API

solve_huber returns HuberResult with:

- solution;
- iteration count;
- convergence flag;
- final Huber objective.

The implementation exposes configuration explicitly through delta, max_iter,
and tol. No hidden global state is used.

## Limitations

The current solver is dense and uses a fixed Huber threshold delta. It does not
yet estimate a robust residual scale, support sparse matrices, or provide
high-breakdown estimators such as least-trimmed squares.

Those are deliberate future extensions, not current capabilities.
