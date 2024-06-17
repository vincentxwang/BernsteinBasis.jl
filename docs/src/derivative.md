# Derivative

We define the 3D Bernstein derivative matrices ``\bm{D}^r,\bm{D}^s, \bm{D}^t`` (synonymous with `BernsteinDerivativeMatrix_3D_r`, `BernsteinDerivativeMatrix_3D_s`, `BernsteinDerivativeMatrix_3D_t` respectively) to be the linear operators that satisfy

```math
\begin{align*}
\sum_{j=1}^{N_p}(\bm{D}^r \bm{p})_j \ \psi_j &= \frac{\partial p}{\partial r} & \sum_{j=1}^{N_p}(\bm{D}^s \bm{p})_j \ \psi_j &= \frac{\partial p}{\partial s} & \sum_{j=1}^{N_p}(\bm{D}^t \bm{p})_j \ \psi_j &= \frac{\partial p}{\partial t}
\end{align*}
```

where ``\bm{p}`` are the coefficients representing a polynomial ``\bm{p} = \sum_{j=1}^{N_p} \bm{p}_j \psi_j``.
