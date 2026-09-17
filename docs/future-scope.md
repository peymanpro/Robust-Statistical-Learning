# Future Scope

This document describes possible post-MVP extensions and long-term directions.

**Critical**: These are exploratory possibilities, **not** current implementation commitments.

The MVP consists of four phases (Least Squares, Regularization, PCA, Real Data Validation). Everything in this document is beyond that scope.

## Numerical Extensions

### Broader Matrix Decompositions

**LU Decomposition**
- PA = LU factorization
- Application: solving square linear systems
- Numerical stability: consider partial/complete pivoting
- Timing: post-MVP if investigating square system methods

**Cholesky Decomposition**
- A = LL^T for positive definite matrices
- Application: efficient solution of symmetric positive definite systems
- Timing: post-MVP if investigating symmetric systems

**Eigenvalue Methods**
- Power iteration
- QR algorithm for eigenvalues
- Lanczos method for large sparse eigenvalue problems
- Timing: post-MVP for theoretical depth

### Iterative Linear Solvers

**Krylov Subspace Methods**
- GMRES (Generalized Minimal Residual)
- CG (Conjugate Gradient)
- MINRES, SYMMLQ for symmetric systems
- BiCG, BiCGSTAB for nonsymmetric systems

**Motivation**: Iterative methods scale better for large sparse systems than direct decompositions.

**Timing**: Post-MVP if investigating large-scale problems.

### Sparse Numerical Methods

**Sparse Storage**
- Compressed sparse row (CSR) format
- Compressed sparse column (CSC) format
- Sparse matrix operations

**Sparse Solvers**
- Sparse QR decomposition
- Sparse Cholesky
- Sparse least squares

**Motivation**: Many real-world problems have sparse structure that dense methods cannot exploit.

**Timing**: Post-MVP if investigating large-scale or industrial applications.

### Additional Stability Analysis

**Error Propagation Analysis**
- Detailed backward error bounds for each algorithm
- Perturbation theory for computed solutions
- Rigorous condition number estimates

**Mixed-Precision Arithmetic**
- Investigate behavior with different floating-point precisions
- Single precision (float32) vs double precision (float64)
- Quadruple precision arithmetic for validation

**Timing**: Post-MVP if building a comprehensive numerical analysis framework.

## Statistical Learning & Optimization

### Optimization Algorithms

**Gradient Descent**
- Basic gradient descent
- Line search strategies
- Convergence analysis

**Newton's Method**
- Newton's method for unconstrained optimization
- Quasi-Newton methods (BFGS, L-BFGS)
- Convergence guarantees

**Coordinate Descent**
- Coordinate descent for convex optimization
- Proximal methods
- Applications to LASSO, elastic net

**Convex Optimization**
- Convex problem formulations
- Convex function properties
- Subgradient descent
- Applications to constrained problems

**Motivation**: Extending from least squares to broader optimization problems.

**Timing**: Post-MVP, potentially alongside a second-phase learning project.

### Regularization Extensions

**Elastic Net**
- Combination of L2 and L1 penalties
- \(\|Ax - b\|^2 + \lambda_2 \|x\|^2 + \lambda_1 \|x\|_1\)
- Parameter selection (grid search, cross-validation)

**LASSO (L1 Regularization)**
- \(\|Ax - b\|^2 + \lambda \|x\|_1\)
- Sparse solutions
- Computational algorithms (coordinate descent, proximal)

**Motivation**: L1 and elastic net provide sparse solutions, useful for feature selection and high-dimensional problems.

**Timing**: Post-MVP if extending to modern statistical learning.

### Cross-Validation

**k-Fold Cross-Validation**
- Standard train/validation split strategy
- Stratified k-fold for classification
- Time-series cross-validation

**Model Selection**
- Grid search for hyperparameter selection
- Randomized search
- Early stopping for iterative methods

**Timing**: Post-MVP, needed for Phase 4 (Real Data Validation).

### Classification Methods

**Logistic Regression**
- Binary classification
- Multinomial extension
- Optimization via gradient descent

**Linear Discriminant Analysis**
- LDA for classification
- Connection to least squares
- Relationship to PCA

**Timing**: Post-MVP if extending to supervised learning beyond regression.

## Deep Learning

**Note**: Deep learning is speculative and would require significant additional development.

### Neural Network Foundations

**Basic Layers**
- Linear/Dense layers
- Activation functions (ReLU, sigmoid, tanh, softmax)
- Batch normalization

**Backpropagation**
- Automatic differentiation
- Gradient computation for multilayer networks
- Chain rule implementation

**Loss Functions**
- Mean squared error
- Cross-entropy loss
- Custom loss formulations

### Optimization for Neural Networks

**Stochastic Gradient Descent (SGD)**
- Mini-batch training
- Learning rate schedules
- Momentum methods (Nesterov)

**Adaptive Methods**
- Adam optimizer
- RMSprop
- AdaDelta

### Regularization for Deep Learning

**Dropout**
- Training with random neuron deactivation
- Inference with scaled output

**Weight Decay**
- L2 regularization
- Early stopping

### Convolutional Neural Networks (CNN)

**Convolutional Layers**
- Convolution operations
- Padding and stride
- Pooling layers

**Applications**
- Image classification
- Feature learning

### Recurrent Neural Networks (RNN)

**LSTM and GRU**
- Long short-term memory
- Gated recurrent units
- Sequence modeling

**Applications**
- Time series prediction
- Natural language processing

### GPU Acceleration

**CUDA Support**
- GPU memory management
- Kernel development
- CuPy integration

**Performance**
- GPU vs CPU comparison
- Parallel algorithm design

**Timing**: Post-MVP, requires significant infrastructure and domain expertise.

## Platform & Deployment

### REST API

**Framework Options**
- FastAPI (modern, async-ready)
- Django REST Framework (mature, feature-rich)
- Flask + extensions (lightweight)

**API Design**
- Endpoints for each learning method
- Input validation
- Error handling
- Response formatting

**Example Endpoints**
```
POST /api/least-squares/solve
  Input: {"A": [...], "b": [...]}
  Output: {"x": [...], "residual": ..., "condition_number": ...}

POST /api/ridge/solve
  Input: {"A": [...], "b": [...], "lambda": ...}
  Output: {"x": [...], ...}

POST /api/pca/fit
  Input: {"X": [...], "n_components": ...}
  Output: {"components": [...], "explained_variance": [...]}
```

### Docker Containerization

**Image Creation**
- Python base image
- Dependencies in requirements.txt or pyproject.toml
- Executable service

**Deployment**
- Single container for testing
- Multi-container with docker-compose
- Kubernetes orchestration (if scaling needed)

### Web Dashboard

**Frontend**
- Interactive parameter input
- Visualization of results
- Experiment history

**Backend Integration**
- REST API communication
- Result caching
- User session management

### Distributed Computation

**Task Distribution**
- Dask for parallel computation
- Ray for distributed machine learning
- Job scheduling

**Applications**
- Large-scale experiments
- Parameter sweep automation
- Cross-validation parallelization

### Microservices Architecture (Speculative)

**Potential Services**
- Solver service (QR, SVD, iterative)
- Experiment orchestration service
- Data pipeline service
- Visualization service

**Communication**
- Message queues (RabbitMQ, Kafka)
- Service discovery
- Load balancing

**Timing**: Post-MVP if the project reaches production scale.

## Compatibility & Reimplementation

### NumPy-Inspired API

**Selected Functions**
- `linalg.qr()` — QR decomposition
- `linalg.svd()` — Singular Value Decomposition
- `linalg.cond()` — Condition number
- `linalg.solve()` — Linear system solver
- `linalg.lstsq()` — Least squares solution

**Motivation**: Familiar API for users already using NumPy.

**Constraint**: NOT a full NumPy reimplementation. Only selected functions, implemented for education and investigation, not performance.

### SciPy-Inspired API

**Selected Functions**
- `linalg.qr()` — with mode options
- `linalg.svd()` — with various options
- `linalg.lstsq()` — robust least squares
- `optimize.linprog()` — linear programming (post-MVP)

**Motivation**: SciPy provides more sophisticated interfaces with options and flexibility.

### Benchmark Comparison

**Performance Tracking**
- Speed comparison with NumPy/SciPy
- Memory usage analysis
- Scalability limits

**Acknowledgment**
> This project will be slower than NumPy/SciPy. Performance is not the goal. NumPy/SciPy are heavily optimized and battle-tested. Use them for production. Use this project to understand the mathematics and numerical behavior.

### Educational Reimplementation

**Purpose**
- Demonstrate algorithm implementation
- Show numerical pitfalls and how to avoid them
- Connect mathematics to code

**Approach**
- Clear, readable code
- Extensive comments explaining numerical decisions
- Comparison to reference implementations
- Pedagogical examples

## Long-Term Research Directions

### Uncertainty Quantification

**Bayesian Methods**
- Posterior distribution over solutions
- Credible intervals for predictions
- Uncertainty propagation

**Applications**
- Decision-making under uncertainty
- Active learning

### Robust Statistics

**Robust Regression**
- Huber loss, absolute deviation loss
- Outlier-resistant methods
- Breakdown point analysis

**Motivation**: Real data often has outliers; ordinary least squares is sensitive to them.

### Matrix Completion

**Problem**
- Recover missing entries in a partially observed matrix
- Applications: recommender systems, sensor networks

**Approaches**
- Nuclear norm minimization
- Alternating least squares
- Probabilistic methods

### Tensor Methods

**Tensor Decompositions**
- CP decomposition (Candecomp/Parafac)
- Tucker decomposition
- Applications to multilinear data

### Machine Learning Theory

**Generalization Bounds**
- PAC learning bounds
- VC dimension analysis
- Margin-based bounds

**Motivation**: Understand when and why learning algorithms generalize.

---

## Guiding Principle

All future work should answer the question:

> **How does mathematics, numerical reliability, or learning behavior change when we add this capability?**

Work that only adds features without deeper understanding is not aligned with the project's purpose.

## Implementation Order (If Pursued)

1. **Numerical Extensions** (LU, Cholesky, basic iterative solvers)
2. **Optimization Algorithms** (gradient descent, Newton, coordinate descent)
3. **Regularization Extensions** (elastic net, LASSO)
4. **Classification** (logistic regression, LDA)
5. **REST API** (to enable broader use)
6. **Deep Learning** (ambitious; requires substantial effort)
7. **Distributed Computation** (if scaling to large problems)
8. **Advanced Topics** (uncertainty quantification, robust statistics, etc.)

## Non-Goals

The following are explicitly **out of scope**, even post-MVP:

- General-purpose NumPy/SciPy replacement
- Production-level performance optimization
- Hardware-specific optimization (GPU CUDA without numerical insight)
- Enterprise features (logging, monitoring, alerts) until there's a production need
- Arbitrary feature requests; all new work must serve the core mission
- Support for old Python versions (target Python 3.12+)
- Compatibility with Java, C++, or other languages (stay focused on Python)
