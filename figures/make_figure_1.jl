#########################################################
#=
Code to compare a local vrs non-local invariant quadratic estimate.
Corresponds to Figures 1 in the paper.
Run this script with the command `include("make_figure_1.jl")` entered into the Julia REPL in this directory.
Run this script from the Julia command line in this directory.
This script saves figure1.pdf to the current directory.
To include figure1.pdf in the paper it must be copied to the directory paper/
=#
#########################################################

using NonstationaryPhase
using PyPlot

savefigures = true # set to `true` if you want the generated figures saved

# --- set grid geometry
const d_const = 1
const period_const = 2π
const nside_const  = 2^12

# ---- Matern parameters for Cϕϕ
const σϕ, ρϕ, νϕ = 0.03, 0.1*2π, 3.0

#  --- Matern parameters for the base spectral density Ck
const σ, ρ, ν = 1, 0.025, 2


####################################
#=
Define two instances of PhaseParms which carries all the parameters 
for the locally stationary model and the non-local-model
=#
#####################################

# ---- local invariant model
parms = PhaseParms(d_const, period_const, nside_const)
parms.ηk[:]       =  parms.k
parms.Ck[:]       =  maternk(parms.k; ν=ν, ρ=ρ, σ=σ)
parms.C1k[:]      =  Array{Complex{Float64}, 1}[ im .* parms.ηk[1] .* parms.Ck ./  (2π) ^ (1/2) ]
parms.C2k[1,1][:] =  -abs2(parms.ηk[1]) .* parms.Ck ./ ((2π)^(1/2)) ./ factorial(2)
parms.Cϕϕk[:]     =  maternk(parms.k; ν=νϕ, ρ=ρϕ, σ=σϕ)
parms.CZZmk[:]    = parms.Ck[:]
parms.ξk[:]       = Array{Complex{Float64},1}[ ones(parms.k[1]) ]  
parms.CZZmobsk[:] = parms.CZZmk + parms.CNNk 
parms.CZZmobsk[parms.r .< 1]              = Inf 
parms.CZZmobsk[parms.r .>= 0.9*parms.nyq] = Inf 

# ---- non local invariant model
parmsNLI          = deepcopy(parms)
parmsNLI.C1k[:]   = map(abs, parms.C1k)
parmsNLI.C2k[:]   = -parms.C2k[:]



####################################
#
#  Bias and variance spectra for local and non-local invariant models
#
#####################################

# ---  approximate variance
varℓ      = (2/√(2π)) .* NonstationaryPhase.local_Aℓ(true,  parms)
tld_varℓ  = (2/√(2π)) .* NonstationaryPhase.local_Aℓ(false, parmsNLI)

# ---  exact second order bias spectrum
θ2biask     = NonstationaryPhase.local_θ2bias(true,  parms)
tld_θ2biask = NonstationaryPhase.local_θ2bias(false, parmsNLI)


####################################
#=
Plot and save
=#
#####################################

fig1 = figure(figsize = (11,4))
    plotrng = 1:200

    # bias, signal and variance for the local invariant model
    f1 = subplot(1,2,1)
    loglog(abs2(parms.k[1][plotrng]) .* parms.Cϕϕk[plotrng], "k:", label = "signal spectral density", linewidth = 3.0)
    loglog(abs2(parms.k[1][plotrng]) .* θ2biask[plotrng]     , "r--",label = "bias spectral density", linewidth = 2.0)
    loglog(abs2(parms.k[1][plotrng]) .* varℓ[plotrng]        , "g",label = "variance spectral density", linewidth = 2.0)
    legend(loc="best",fontsize = 10)
    xlabel("frequency")
    ylabel("spectral density")
    title("Local invariant model")

    # bias, signal and variance for the non-local-invariant model
    f2 = subplot(1,2,2, sharex=f1, sharey=f1)
    loglog(abs2(parms.k[1][plotrng]) .* parms.Cϕϕk[plotrng], "k:"  , label = "signal spectral density", linewidth = 3.0)
    loglog(abs2(parms.k[1][plotrng]) .* tld_θ2biask[plotrng] , "r--", label = "bias spectral density", linewidth = 2.0)
    loglog(abs2(parms.k[1][plotrng]) .* tld_varℓ[plotrng]    , "g",label = "variance spectral density", linewidth = 2.0)
    xlabel("frequency")
    ylabel("spectral density")
    title("Non local invariant model")
    axis("tight")


if savefigures

    fig1[:savefig]("figure1.pdf", dpi=300, bbox_inches="tight", transparent=true)
    plt[:close](fig1)

end 



#= #######################################################
---- Take a look at the weights for both models
---- Notice that the non-local invariant models puts too much emphasis on a few high frequency terms.
-----In contrast the local invariant weights are nearly flat over a wide region 
-----(this which improves Gaussianity of the estimate, for one thing)
-----Not included in the paper. Saved for reference later.
=# 
if false
   function getweights(islocvar::Bool, ell, parms)
        mlt = islocvar ? -1 : 1 
        δ_0 = 1/(parms.deltk ^ 1)
        # locally stationary weights w_locst_k
        Aℓ = NonstationaryPhase.local_Aℓ(islocvar, parms)
        C1k_ell = NonstationaryPhase.shiftfk_by_ω(parms.C1k[1], ell, parms)
        CZZmobsk_ell = circshift(parms.CZZmobsk, -ell)
        w_locst_k   = abs2( parms.ξk[1][ell].*(parms.C1k[1] + mlt * C1k_ell) )
        w_locst_k ./= parms.CZZmobsk
        w_locst_k ./= CZZmobsk_ell
        w_locst_k .*= real(Aℓ[ell])
        return w_locst_k
    end
    δ_0 = 1/(parms.deltk ^ 1)
    weightsk     = getweights(true,  30, parms)
    weightsNLIk  = getweights(false, 30, parmsNLI)
    figure()
    semilogy(parms.k[1]|> fftshift, real(weightsk) |> fftshift, label="locally invariant stationary weights")
    semilogy(parms.k[1]|> fftshift, real(weightsNLIk) |> fftshift, label="non locally invariant weights")
    axis("tight")
    @show sum(δ_0 .* weightsk ./ ((2π)^(1/2))  )
    @show sum(δ_0 .* weightsNLIk ./ ((2π)^(1/2))  )
end



