# Notation and conventions

## Reference tetrahedron

We define our reference tetrahedron ``\hat{D}`` in ``rst``-space by
```math
\hat{D} := \{-1 \leq r,s,t \mid r + s + t \leq 1\}.
```

We define the barycentric coordinates ``\lambda_0, \lambda_1, \lambda_2, \lambda_3`` on ``\hat{D}`` by

```math
\begin{aligned}
\lambda_0 &:= \frac{1 + r}{2} & \lambda_1 &:= \frac{1 + s}{2} & \lambda_2 &:= \frac{1 + t}{2} & \lambda_3 &:= - \frac{1 + r + s + t}{2}. \\
\end{aligned}
```

## Bernstein basis

Consider 4-tuples of non-negative integers ``(i, j, k, l)`` satisfying ``i + j + k + l = N``, where ``N`` is the desired degree of the Bernstein basis. Using these tuples, we define the ``(i, j, k, l)``-th three-dimensional Bernstein basis function $\psi_{(i,j,k,l)}$ as
```math
\psi_{(i, j, k, l)} := \frac{N!}{i!j!k!l!} \lambda_0^i \lambda_1^j \lambda_2^k \lambda_3^l.
```

Note that there exist 

```math
N_p = \frac{(N + 1)(N + 2)(N + 3)}{6}
```
of these tuples for an ``N``-degree 3D Bernstein basis. 

We will also assign each basis function a scalar index ``n`` by ordering the *first three* coordinates in reverse dictionary order. The `ordering.jl` file contains tools for efficiently converting between the two.



