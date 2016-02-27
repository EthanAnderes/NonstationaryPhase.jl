# NonstationaryPhase

[![Build Status](https://travis-ci.org/EthanAnderes/NonstationaryPhase.jl.svg?branch=master)](https://travis-ci.org/EthanAnderes/NonstationaryPhase.jl)

This is a Julia package which holds all the source code used in the paper "A generalized quadratic estimate for random field nonstationarity" by Ethan Anderes (University of California at Davis) and Joe Guinness (North Carolina State University). 

All the code was run on Julia 0.4 (installation instructions can be found at [julialang.org](http://julialang.org)). The exact version information for the system which generated the figures are given as follows. 

```julia
julia> Pkg.installed("PyPlot")
v"2.1.1"

julia> Pkg.installed("Distributions")
v"0.8.10"

julia> versioninfo()
Julia Version 0.4.4-pre+29
Commit d2a82a0 (2016-02-09 14:59 UTC)
Platform Info:
  System: Darwin (x86_64-apple-darwin15.3.0)
  CPU: Intel(R) Core(TM) i7-4850HQ CPU @ 2.30GHz
  WORD_SIZE: 64
  BLAS: libopenblas (USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell)
  LAPACK: libopenblas64_
  LIBM: libopenlibm
  LLVM: libLLVM-3.3
```

This information is intended to enable anyone to recreate the exact system which generated the figures in the paper. That said, we expect the code to be forward compatible with stable releases.

# Using Julia to automatically download the source code and the paper from Github

The following code uses the package manager in Julia to download and install the package `NonstationaryPhase`. 

```julia
julia> Pkg.clone("https://github.com/EthanAnderes/NonstationaryPhase.jl.git")
```

Note: `julia>` above indicates that the code which follows should be executed in the Julia REPL (i.e. the Julia command line). Typically the package is downloaded to `~/.julia/v0.4/`. You can use Julia to tell you where the package is with the command `Pkg.dir("NonstationaryPhase")` in the Julia REPL. 

Now use the following code to generate Figure 1 (for example) from the paper.

```julia
julia> using NonstationaryPhase 
julia> cd(joinpath(Pkg.dir("NonstationaryPhase"),"figures"))
julia> include("make_figure_1.jl") 
```

This will save `figure1.pdf` to the current directory.

To remove this package you can either directly delete the package directory (e.g. use something like `rm -fr ~/.julia/v0.4/NonstationaryPhase` in the terminal). Or you can use the Julia package manager to remove it for you with the command `Pkg.rm("NonstationaryPhase")`.


# Clone the package directly from github.

If you want to avoid using the Julia package manager you can use the following code to download git repo `NonstationaryPhase` manually from the terminal.

```
$ git clone https://github.com/EthanAnderes/NonstationaryPhase.jl.git
```

Now you need too make sure Julia knows where to find `NonstationaryPhase.jl`. 


```julia
julia> push!(LOAD_PATH, "<path to the directory containing NonstationaryPhase.jl>")
```

Now you should be able to run the code for Figure 1 (for example) as follows.

```julia

julia> using NonstationaryPhase
julia> cd("<path to NonstationaryPhase.jl/figures>")
julia> include("make_figure_1.jl")
```

This will save `figure1.pdf` to the current directory.
