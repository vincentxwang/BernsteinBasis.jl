# BernsteinBasis

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://vincentxwang.github.io/BernsteinBasis.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vincentxwang.github.io/BernsteinBasis.jl/dev/)
[![Build Status](https://github.com/vincentxwang/BernsteinBasis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vincentxwang/BernsteinBasis.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/vincentxwang/BernsteinBasis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/vincentxwang/BernsteinBasis.jl)

[This package](https://github.com/vincentxwang/BernsteinBasis.jl) contains performant reference element operators and algorithms for using a Bernstein basis in discontinuous Galerkin methods. The codes are based on [GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems](https://arxiv.org/abs/1512.06025) by Chan and Warburton (2015). Examples of using this package in DG solvers can be found in `/examples`.