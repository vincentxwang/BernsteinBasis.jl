# Derivative

We define the 3D Bernstein derivative matrices ``\bm{D}^r,\bm{D}^s, \bm{D}^t`` (synonymous with `BernsteinDerivativeMatrix_3D_r`, `BernsteinDerivativeMatrix_3D_s`, `BernsteinDerivativeMatrix_3D_t` respectively) to be the linear operators that satisfy

```math
\begin{align*}
\sum_{j=1}^{N_p}(\bm{D}^r \bm{p})_j \ \psi_j &= \frac{\partial p}{\partial r} & \sum_{j=1}^{N_p}(\bm{D}^s \bm{p})_j \ \psi_j &= \frac{\partial p}{\partial s} & \sum_{j=1}^{N_p}(\bm{D}^t \bm{p})_j \ \psi_j &= \frac{\partial p}{\partial t}
\end{align*}
```

where ``\bm{p}`` are the coefficients representing a polynomial ``p = \sum_{j=1}^{N_p} \bm{p}_j \psi_j``.

In contrast to a nodal scheme which has a dense ``N_p \times N_p`` matrix that multiplies a vector in ``O(N_p^2) ~ O(N^6)``, we exploit sparsity in the Bernstein derivative matrix to multiply a vector in optimal ``O(N^3)`` time complexity.