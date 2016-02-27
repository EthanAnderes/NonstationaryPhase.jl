###################################
#=
Simple implementations, for dimension 1, of the quadratic estimate, Cvar and Cbias which can be used
to generate estimates for non-local invariant models. Used in `make_figure_a.jl` and `make_figure_b.jl`

=#
######################################


function local_Aℓ(islocvar::Bool, parms)
    ξk, C1k, CZZmobsk = parms.ξk, parms.C1k, parms.CZZmobsk
    deltx, deltk = parms.deltx, parms.deltk
    mlt = islocvar ? -1 : 1 
    CXXinvCZZ = squash(1 ./ CZZmobsk)
    Apqx = ifftd( C1k[1] .* conj(C1k[1]) .* CXXinvCZZ, deltk)
    Bx   = ifftd(                    1.0 .* CXXinvCZZ, deltk)
    Cpx  = ifftd(                 C1k[1] .* CXXinvCZZ, deltk)
    Dpx  = ifftd(           conj(C1k[1]) .* CXXinvCZZ, deltk)
    ABCDx = zeros(Complex{Float64}, size(CZZmobsk))
    for ix in eachindex(ABCDx)
        ABCDx[ix] = 2 * Apqx[ix] * Bx[ix] + mlt * Cpx[ix] * Cpx[ix] + mlt * Dpx[ix] * Dpx[ix]
    end
    rtnk = 1./ abs2(ξk[1]) ./ real(fftd(ABCDx, deltx))
    squash!(rtnk)
    return rtnk
end
#= ---- Basic Test -----
include("make_figure_a.jl",1:53)
test1 = (2/√(2π)) .* NonstationaryPhase.local_Aℓ(true, parms) 
test2 = (2/√(2π)) ./ invAℓfun(parms.CZZmobsk, parms)
test3 = Cℓvarfun(parms)

test1./test2 # should be [*,1,1 ..., 1] 
test1./test3 # should be [*,1,1 ..., 1] 
=#


function local_hatθfun(islocvar::Bool, xk, yk, Aell, parms)
    ξk, C1k, CZZmobsk = parms.ξk, parms.C1k, parms.CZZmobsk
    deltx, deltk = parms.deltx, parms.deltk
    mlt = islocvar ? -1 : 1 
    xkinvCZZ = xk ./ CZZmobsk
    ykinvCZZ = yk ./ CZZmobsk
    squash!(xkinvCZZ)
    squash!(ykinvCZZ)
    A1x  = ifftd(                 xkinvCZZ, deltk)
    Dpx  = ifftd(      C1k[1]  .* ykinvCZZ, deltk)
    A2x  = ifftd(                 ykinvCZZ, deltk)
    Cpx  = ifftd( conj(C1k[1]) .* xkinvCZZ, deltk)
    rtnk   = conj(ξk[1]) .* fftd(A1x.*Dpx + mlt.*A2x.*Cpx, deltx)
    rtnk .*= Aell
    squash!(rtnk)
    return rtnk
end
#= ---- Test local_hatθfun on locally invariant models -----
include("make_figure_a.jl",1:53)
Aelltest = NonstationaryPhase.local_Aℓ(true, parms) 
xk = randn(parms.nside) + im * randn(parms.nside)
yk = randn(parms.nside) + im * randn(parms.nside)
test1 = NonstationaryPhase.local_hatθfun(true, xk, yk, Aelltest, parms)
test2 = let 
    rtnk   = unnormalized_estϕkfun(xk, yk, parms)
    rtnk ./= invAℓfun(parms)
    squash!(rtnk)
    rtnk
end

real(test1)./real(test2)  # should be [*,1,1 ..., 1] 
imag(test1)./imag(test2)  # should be [*,1,1 ..., 1] 
=#


# ---- Test local_hatθfun on nonlocally invariant models -----
function test_local_hatθfun(sims, ϕx, parmsNLI) 
    dm     = ndims(parmsNLI.x)
    Aell   = NonstationaryPhase.local_Aℓ(false, parmsNLI) 
    Eestϕk = zeros(Complex{Float64}, size(ϕx))
    for iter = 1:sims
        wk     =  fftd(randn(parmsNLI.nside)./√(parmsNLI.deltx ^ dm), parmsNLI.deltx)
        xk     = √(parmsNLI.Ck) .* wk
        yktmp  = √(2π) * (parmsNLI.C1k[1,1] ./ √(parmsNLI.Ck)) .* wk
        yx     = ϕx .* ifftd(yktmp, parmsNLI.deltk)
        yk     = fftd(yx, parmsNLI.deltx)
        estϕk  = NonstationaryPhase.local_hatθfun(false, xk, yk, Aell, parmsNLI)
        estϕk  += NonstationaryPhase.local_hatθfun(false, yk, xk, Aell, parmsNLI)
        Eestϕk += estϕk ./ sims
    end
    Eestϕx = ifftdr(Eestϕk, parmsNLI.deltk)
    return Eestϕx, Eestϕk
end
#= ----- this runs an example of the test using test_local_hatθfun ----
    using NPhaseGRF
    using PyPlot    
    const d_const      = 1
    const period_const = 2π
    const nside_const  = nextprod([2,3,5,7], 2_500) 
    ν  = 3.0 
    ρ  = 0.02 
    νΦ = 5.0
    ρΦ = 0.5
    σΦ = 0.002
    parmsNLI               = PhaseParms(d_const, period_const, nside_const)
    parmsNLI.Ck[:]         =   maternk(parmsNLI.k; ν=ν, ρ=ρ)
    parmsNLI.C1k[1,1][:]   = parmsNLI.Ck .* (2 - abs2(parmsNLI.r.*ρ)) ./ (4ν + abs2(parmsNLI.r.*ρ))
    parmsNLI.C1k[1,1][:] .*= (2ν / ρ) / (2*√(2π))
    parmsNLI.ξk[:]         = Array{Complex{Float64},1}[ ones(parmsNLI.k[1]) ]  
    parmsNLI.Cϕϕk[:]      =  maternk(parmsNLI.k; ν=νΦ, ρ=ρΦ, σ=σΦ)
    parmsNLI.CZZmk[:]      =  parmsNLI.Ck[:]
    parmsNLI.CZZmobsk[:]   = parmsNLI.CZZmk + parmsNLI.CNNk 
    parmsNLI.CZZmobsk[parmsNLI.r .< 1]                 = Inf 
    parmsNLI.CZZmobsk[parmsNLI.r .>= 0.9*parmsNLI.nyq] = Inf 
  
    ϕx = NonstationaryPhase.grfsimx(parmsNLI.Cϕϕk, parmsNLI.deltx, parmsNLI.deltk)
    ϕx += -minimum(real(ϕx)) + ρ

    Eestϕx, Eestϕk = NonstationaryPhase.test_local_hatθfun(5000, ϕx, parmsNLI) 

    plot(fftshift(Eestϕx), "k", alpha = 0.4)
    plot(fftshift(ϕx), "b")
=#


function local_θ2bias(islocvar::Bool, parms)
    ξk, CZZmobsk, C2k  = parms.ξk,  parms.CZZmobsk, parms.C2k[1,1] 
    deltx, deltk, xco, kco = parms.deltx, parms.deltk, parms.x, parms.k
    m1s   = ones(parms.Cϕϕk)
    mlt = islocvar ? -1 : 1 
    Aell = local_Aℓ(islocvar, parms)
    hatθC2_kplusℓ  =  local_hatθfun(islocvar, C2k, m1s, Aell, parms)
    rtnk     = zeros(Float64, size(CZZmobsk))
    tmp_rtnk = zeros(Float64, size(CZZmobsk))
    for ωindx in eachindex(kco[1])
        ω = kco[1][ωindx]
        hatθC2_kplusℓminusω =  local_hatθfun(islocvar, shiftfk_by_ω(C2k, -ω, parms), m1s, Aell, parms)
        tmp_rtnk   =  abs2(hatθC2_kplusℓ + mlt.*hatθC2_kplusℓminusω)
        tmp_rtnk .*=  shiftfk_by_ω(ξk[1] .* conj(ξk[1]) .* parms.Cϕϕk, -ω, parms)
        tmp_rtnk .*=  ξk[1][ωindx] * conj(ξk[1][ωindx]) * parms.Cϕϕk[ωindx] 
        rtnk     .+=  tmp_rtnk
    end
    rtnk .*= 2 * 4 * (deltk / 2π)  
    # the extra 2 in the above constant is because in 1-d, the two terms involving Cθθk.*Cθθk_kminusω are the same.
    return rtnk
end
#= ---- Basic Test -----
include("make_figure_a.jl",1:53)
Cℓbias_test1 = NonstationaryPhase.local_θ2bias(true,  parms)
Cℓbias_test2 = approxCℓbiasfun(parms)
Cℓbias_test3 = Cℓbiasfun(parms)
Cℓbias_test1 ./ Cℓbias_test3   # should be [1,1,1 ..., 1] 
Cℓbias_test2 ./ Cℓbias_test3   # should be [1,1,1 ..., 1] 

=#


