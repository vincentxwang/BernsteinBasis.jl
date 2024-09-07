# Lift

For a face ``f``, we define the lift matrix ``\mathbf{L}^f`` to be

```math
\mathbf{L}^f = \mathbf{M}^{-1}\mathbf{M}^f,
```
where ``\mathbf{M}`` and ``\mathbf{M}^f`` are the Bernstein mass matrices defined by

```math
\begin{aligned}
\mathbf{M}_{i,j} &= \int_{\hat{D}} \phi_{i} \phi_{j} \\
\mathbf{M}_{i,j}^f &= \int_{f_{\hat{D}}} \phi_{i} \varphi_{j}. \\
\end{aligned}
```

where ``\{\varphi_j\}_{j=1}^{j=\frac{N(N+1)}{2}}`` is the subset of ``\{\phi_j\}_{j=1}^{j=\frac{N(N+1)(N+2)}{6}}`` that is *not* constantly zero on ``f``, or the trace of the polynomial space.

We define ``\mathbf{L}`` (synonymous with `BernsteinLift`) to be

```math
\mathbf{L} = \left[ \begin{array}{c|c|c|c} \mathbf{L}^1 & \mathbf{L}^2 & \mathbf{L}^3 & \mathbf{L}^4 \end{array} \right].
```

This `BernsteinLift` implementation transforms a vector in an optimal ``O(N^3)``, improving upon the typical ``O(N^5)`` time complexity of multiplying a dense nodal lift matrix by a vector.