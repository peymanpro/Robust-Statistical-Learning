# Error Analysis

The project distinguishes three questions about a computed least-squares
solution:

1. Forward error — how far is the computed parameter vector from the
   reference solution?
2. Backward error — how well does the computed vector satisfy a nearby
   least-squares problem?
3. Residual orthogonality — how closely does the computed vector satisfy
   the first-order least-squares condition A^T(Ax-b)=0?

## Forward Error

For a computed vector x_hat and a reference x_star:

    forward(x_hat) = ||x_hat - x_star||_2 / ||x_star||_2.

This directly measures solution accuracy. It is the quantity that can become
large even when the residual is tiny on an ill-conditioned problem.

## Normwise Backward-Error Indicator

The project reports:

    eta = ||A x_hat - b||_2 /
          (||A||_F ||x_hat||_2 + ||b||_2).

This is a residual-based, normwise indicator. A small value says that the
computed solution nearly satisfies the original equations after a small
relative perturbation of the data. It should not be interpreted as the exact
minimum backward error under every possible perturbation model.

## Residual Orthogonality

For least squares, the normality condition is:

    A^T(A x_hat - b) = 0.

The implementation reports:

    ||A^T r||_2 / (||A||_2 ||r||_2),  r = A x_hat - b,

with the zero-residual case defined as zero.

QR and SVD should keep this diagnostic near the floating-point rounding scale
on stable full-rank problems. A normal-equations solution can also have a small
residual while exhibiting a larger forward error because forming A^T A squares
the condition number.

## Reproducible Experiment

Run:

    python experiments/error_analysis.py

The experiment fixes the random seed and reports all three error notions for
Normal Equations, QR, and SVD on the same controlled matrix.

The experiment is diagnostic rather than a universal benchmark. Its numbers
depend on matrix dimensions, target condition number, BLAS/LAPACK implementation,
and floating-point environment.

## Interpretation Rule

Never use residual size alone as evidence that a least-squares parameter vector
is accurate. Read residual, forward error, conditioning, and backward-error
indicators together.
