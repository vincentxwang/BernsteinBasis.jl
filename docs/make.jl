using BernsteinBasis
using Documenter

DocMeta.setdocmeta!(BernsteinBasis, :DocTestSetup, :(using BernsteinBasis); recursive=true)

makedocs(;
    modules=[BernsteinBasis],
    authors="Vincent X. Wang <vw12@rice.edu>",
    sitename="BernsteinBasis.jl",
    format=Documenter.HTML(;
        canonical="https://vincentxwang.github.io/BernsteinBasis.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Notations and conventions" => "notation.md",
        "Operators" => [
            "Derivative" => "derivative.md"
            "Lift" => "lift.md"
        ],
        "Full API" => "reference.md",
        "Misc. notes and such" => [
            "Polynomial representation" => "polynomial.md"
        ]
    ],
)

deploydocs(;
    repo="github.com/vincentxwang/BernsteinBasis.jl",
    devbranch="main",
)
