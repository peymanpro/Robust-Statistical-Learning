# Regularization

The project treats regularization as both a statistical and numerical tool.

## Ridge

Ridge solves

    minimize_x ||Ax - b||_2^2 + alpha ||x||_2^2,

for alpha > 0.

For A = U Sigma V^T:

    x_alpha =
        V diag(sigma_i / (sigma_i^2 + alpha)) U^T b.

The implementation evaluates this SVD filter directly. It does not construct
A.T A.

The filter also makes the bias mechanism visible: directions associated with
small singular values receive stronger shrinkage.

## Generalized Tikhonov

For a regularization matrix L:

    minimize_x ||Ax - b||_2^2 + alpha ||Lx||_2^2.

The implementation forms the augmented system

    [ A             ] x ~= [ b ]
    [ sqrt(alpha) L ]     [ 0 ]

and solves it through the existing SVD least-squares path.

This is a generalization of Ridge rather than a separate numerical stack.

## Generalized Cross-Validation

For Ridge, the effective degrees of freedom are

    df(alpha) = sum_i sigma_i^2 / (sigma_i^2 + alpha).

The project reports

    GCV(alpha) = ||A x_alpha - b||_2^2 / (m - df(alpha))^2.

A caller supplies the candidate grid because the useful alpha scale is
problem-dependent. The selector chooses the candidate with the smallest GCV
score and does not pretend that one universal alpha exists.

## Statistical vs Numerical Trade-off

Increasing alpha can:

- reduce variance caused by noise;
- reduce sensitivity in weak singular directions;
- improve effective numerical conditioning;
- increase estimator bias.

The experiments and tests should evaluate those effects separately rather
than treating lower residual as universally better.

## API

The learning layer exposes:

- solve_ridge
- solve_tikhonov
- ridge_gcv_score
- select_ridge_alpha_gcv

## Scope

L1/Elastic Net and iterative proximal methods remain separate future work.
