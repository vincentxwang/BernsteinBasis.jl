# Polynomial representations

Suppose we would like to model a function ``u(\bf{x})`` with a polynomial basis ``\{\psi_i\}_{i=1}^{N_p}``. Assume that ``u`` is in this polynomial space.

One way we can represent ``u`` is by expressing it as a direct sum of the elements of the basis and taking the coefficients. In particular,

```math
u = \sum_{j=1}^{N_p} \hat{\bf{u}}_j \psi_j
```

where ``\hat{\bf{u}}`` is a vector of coefficients. This is called a "modal" representation. It is possible to find these ``\hat{\bf{u}}`` through an ``L^2`` projection of ``u`` onto the basis vectors, but this is rather difficult.

The second way to represent ``u`` is to pick any ``N_p`` points ``\{\xi_i\}_{i=1}^{N_p}``and evaluate ``u`` at these points. We denote the vector ``\bf{u}`` as the "nodal" representaiton, where ``\bf{u}_{\textit{i}} = \textit{u}(\xi_{\textit{i}})``. This provides just as much information as the first approach -- we could solve for the ``N_p`` number of coefficients by writing out all ``N_p``. 

However, there is an easier way to convert between the two. Let the generalized Vandermonde matrix ``\mathcal{V}`` (of size ``N_p \times N_p``) be defined by ``\mathcal{V}_{ij} = \psi_{j}(\xi_i)``. Then it follows that 

```math
(\mathcal{V}\hat{\mathbf{u}})_{i} = \sum_{i=1}^{N_{p}}\hat{\mathbf{u}}_{j}\psi_{j}(x_{i})=u(x_{i})
```

and we can use ``\mathcal{V}`` to transform ``\hat{\bf{u}}`` into ``\bf{u}`` by

```math
\mathcal{V} \hat{\bf{u}} = \bf{u}.
```

# Package

The operators this packages provides are meant to be used with a *modal* representation. While the [derivatives matrices](derivative.md) work with both the nodal and modal representations, the [lift matrix](lift.md) here is only compatable with a modal representation.

# Reference

Jan S Hesthaven and Tim Warburton. Nodal discontinuous Galerkin methods: algorithms, analysis, and applications, volume 54. Springer, 2007.