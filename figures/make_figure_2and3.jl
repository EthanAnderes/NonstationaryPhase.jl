#########################################################
#=
Generate figures for the 1-d nonstationary phase example.
Corresponds to Figures 2 and 3 in the paper.
Run this script with the command `include("make_figure_2and3.jl")` entered into the Julia REPL in this directory.
This script saves figure2.pdf and figure3.pdf to the current directory.
To include  figure2.pdf and figure3.pdf in the paper it must be copied to the directory paper/
=#
#########################################################


# --- load modules
using NonstationaryPhase
using PyPlot

# --- set the seed
#seedstart = rand(UInt64)
seedstart = 0xe8bac1438a62066d
srand(seedstart)

# --- save figures to disk or not
savefigures = true

# --- set grid geometry
const d_const = 1
const period_const = 10
const nside_const  = nextprod([2,3,5,7], 10_000) 

# --- parameters of Z(x)
ν      = 2.0
tild_ν = 2.1 
ρ      = 0.005*period_const
tild_ρ = 0.005*period_const 
t_0    = 1.5

# --- noise parameters
σpixl    = 0.0
beamFWHM = 0.0

# --- prior parameters
νΦ = 5.0
ρΦ = 0.15*period_const
σΦ = 0.15*(period_const/2π)^2 



#########################################################
#=
Generate the first set of simulations
=#
#########################################################


# --- define an instance of PhaseParms which carries all the parameters
parms = PhaseParms(d_const, period_const, nside_const)
ηkfun, Ckfun, C1kfun, C2kfun = gen_tangentMatern(ν=ν, tild_ν=tild_ν, ρ=ρ, tild_ρ=tild_ρ, t_0=t_0)
NonstationaryPhase.CNNkfun{dm,T1,T2}(p::PhaseParms{dm,T1,T2})   = CNNkfun(p.k, p.deltx; σpixl=σpixl, beamFWHM=beamFWHM)
Cϕϕkfun{dm,T1,T2}(p::PhaseParms{dm,T1,T2})  = maternk(p.k; ν=νΦ, ρ=ρΦ, σ=σΦ)
parms.ηk[:]       =  ηkfun(parms.k)
parms.Ck[:]       =  Ckfun(parms.k)
parms.C1k[:]      =  C1kfun(parms.k)
parms.C2k[:]      =  C2kfun(parms.k)
parms.CNNk[:]     =  CNNkfun(parms)
parms.Cϕϕk[:]     = Cϕϕkfun(parms)
parms.CZZmk[:]    =  CZZmkfun(parms.Cϕϕk, parms.Ck, parms.ηk, parms.ξk, parms.x, parms.k, parms.deltx, parms.deltk)
parms.CZZmobsk[:] = parms.CZZmk + parms.CNNk 
parms.CZZmobsk[parms.r .< 1]              = Inf 
parms.CZZmobsk[parms.r .>= 0.9*parms.nyq] = Inf 

# Now a High resolution version of PhaseParms for simulation
parmsHR = PhaseParms(d_const, period_const, nside_const)
parmsHR.ηk[:]   =  ηkfun(parmsHR.k)
parmsHR.Ck[:]   =  Ckfun(parmsHR.k)
parmsHR.CNNk[:] =  CNNkfun(parmsHR)
parmsHR.Cϕϕk[:] =  Cϕϕkfun(parmsHR)


# ---- generate the initial simulation and other plotting quantities
zkobs, zk, zx, zx_noϕ, ϕk, ϕx = simNPhaseGRF(parms, parmsHR)
Dθx          = NonstationaryPhase.ϕx_2_Dθx(ϕx, parms) 
@time Cℓvar        = Cℓvarfun(parms)
@time Cℓbias       = Cℓbiasfun(parms)
@time approxCℓbias = approxCℓbiasfun(parms)
estϕk        = estϕkfun(zkobs, parms)
# plot(zx)
# plot(NonstationaryPhase.ϕk_2_Dθx(estϕk.* (0 .< parms.r .< 20*parms.deltk), parms),"k")
# plot(NonstationaryPhase.ϕk_2_Dθx(ϕk, parms),"b")


# --- get multiple quadratic estimates based on the same ϕx and the average estimate
function get_estimates{dm,T1,T2}(sims::Integer, ϕx, parms::PhaseParms{dm,T1,T2}, parmsHR::PhaseParms{dm,T1,T2}) 
	Eestϕk        = zeros(Complex{Float64}, size(parms.k[1],1))
	estimates_Dθx = zeros(Complex{Float64}, size(parms.k[1],1),sims)
	avetime = 0.0
	for loop = 1:sims
		_zkobs, = simNPhaseGRF(ϕx, parms, parmsHR)
		tic()
		_estϕk  = estϕkfun(_zkobs, parms)
		avetime += toc() / sims
		_estϕx  = ifftdr(_estϕk .* (0 .< parms.r .< 20*parms.deltk), parms.deltk)
		estimates_Dθx[:,loop] = NonstationaryPhase.ϕx_2_Dθx(_estϕx, parms) 
		Eestϕk += _estϕk .* (1/sims) .* (0 .< parms.r .< 40*parms.deltk)
	end
	EestDθx = NonstationaryPhase.ϕx_2_Dθx(ifftdr(Eestϕk,parms.deltk), parms)
	return EestDθx, Eestϕk, estimates_Dθx, avetime
end
EestDθx, Eestϕk, estimates_Dθx, avetime1 = get_estimates(100, ϕx, parms, parmsHR) 
println("ave quadratic estimate clock speed =  $avetime1")



#########################################################
#=
Generate plots for the first set of simulations
=#
#########################################################

# ------ figure ---- show the quadratic estimate 
function makeplots()
	global fig1 = figure(figsize = (10,20))

	# ----- figure: plot Z(x)
	ax1 = subplot2grid((3,2), (0,0), colspan=2)
	plot(fftshift(parms.x[1]), fftshift(zx), "k", linewidth = 1.5)
	axis("tight")
	title("Nonstationary phase simulation")

	# ----- figure: plot the estimates
	ax2 = subplot2grid((3,2), (1,0), colspan=2)
	plot(fftshift(parms.x[1]), fftshift(estimates_Dθx[:,1]),"k", label = L"\hat\theta^\prime(x)", linewidth = 1.5, alpha = 0.4)
	for cntr = 2:min(size(estimates_Dθx,2),5)
		plot(fftshift(parms.x[1]), fftshift(estimates_Dθx[:,cntr]),"k", linewidth = 1.5, alpha = 0.4)
	end
	plot(fftshift(parms.x[1]), fftshift(EestDθx), "k--", label = L"average $\hat\theta^\prime(x)$", linewidth = 3)
	plot(fftshift(parms.x[1]), fftshift(Dθx), "b", label = L"\theta^\prime(x)", linewidth = 2.5,alpha = 0.7)
	axis("tight")
	title(L"Quadratic estimate, average quadratic estimate and the simulation truth $\theta^\prime(x)$")
	legend(loc="best",fontsize = 10)
	
	# ----- figure ---- Cℓbias and the signal
	ax4 = subplot2grid((3,2), (2,1))
	δ0 = 1 / parms.deltk ^ d_const
	kbins, bias_radpower        = radial_power(abs2(parms.r.*(Eestϕk - ϕk))./δ0, 1, parms)
	kbins, Cbias_radpower       = radial_power(abs2(parms.r).*Cℓbias, 1, parms)
	kbins, approxCbias_radpower = radial_power(abs2(parms.r).*approxCℓbias, 1, parms)
	kbins, CΦΦ_radpower         = radial_power(abs2(parms.r).*parms.Cϕϕk, 1, parms)
	semilogy(kbins[1:25], Cbias_radpower[1:25], "r-", label = L"\ell^2 C^{bias\, \hat\phi}_\ell", linewidth = 2.0)
	semilogy(kbins[1:25], approxCbias_radpower[1:25], "b--", label = L"approx $\ell^2 C^{bias\, \hat\phi}_\ell$", linewidth = 2.0)
	semilogy(kbins[1:25], CΦΦ_radpower[1:25], "k:", label = L"\ell^2C^{\phi\phi}_\ell", linewidth = 3.0)
	semilogy(kbins[1:25], bias_radpower[1:25], "ro", label = L"\ell^2|\phi_\ell - E(\hat\phi_\ell)|^2\delta_0", linewidth = 3.0)
	xlabel(L"frequency $\ell$")
	title("Analytic vrs empirical bias spectra")
	legend(loc="best",fontsize = 10)

	# ----- figure ---- Cℓvar and the signal
	ax3 = subplot2grid((3,2), (2,0), sharex=ax4, sharey=ax4)
	δ0 = 1 / parms.deltk ^ d_const
	kbins, var_radpower  = radial_power(abs2(parms.r.*(Eestϕk - estϕk))./δ0, 1, parms)
	kbins, Cvar_radpower = radial_power(abs2(parms.r).*Cℓvar, 1, parms)
	kbins, CΦΦ_radpower  = radial_power(abs2(parms.r).*parms.Cϕϕk, 1, parms)
	semilogy(kbins[1:25], Cvar_radpower[1:25], "g-", label = L"\ell^2 C^{var\, \hat\phi}_\ell", linewidth = 2.0)
	semilogy(kbins[1:25], CΦΦ_radpower[1:25], "k:", label = L"\ell^2C^{\phi\phi}_\ell", linewidth = 3.0)
	semilogy(kbins[1:25], var_radpower[1:25], "go", label = L"\ell^2|E(\hat\phi_\ell) - \hat\phi_\ell|^2\delta_0", linewidth = 3.0)
	ylabel("spectral density")
	xlabel(L"frequency $\ell$")
	title("Analytic vrs empirical variance spectra")
	legend(loc="best",fontsize = 10)
	axis("tight")
	
	#ax4[:set_yticklabels]([])
	fig1[:subplots_adjust](wspace=0.2)

end

# --- now evaluate the above `script'
makeplots()

# --- save the figures to disk
if savefigures
    fig1[:savefig]("figure2.pdf", dpi=300, bbox_inches="tight", transparent=true)
    plt[:close](fig1)
end 

##############################################################
#=
Compute the minimum cut loci and increase the signal size to bring out the bias
=#
##############################################################

# --- reset the seed to reproduce the same results as above
srand(seedstart)

# --- now decrease t_0 to expose the bias
reduce_t_0_factor = 1/7

# --- define an instance of PhaseParms which carries all the parameters
parms = PhaseParms(d_const, period_const, nside_const)
ηkfun, Ckfun, C1kfun, C2kfun = gen_tangentMatern(ν=ν, tild_ν=tild_ν, ρ=ρ, tild_ρ=tild_ρ, t_0=t_0*reduce_t_0_factor)
NonstationaryPhase.CNNkfun{dm,T1,T2}(p::PhaseParms{dm,T1,T2})   = CNNkfun(p.k, p.deltx; σpixl=σpixl, beamFWHM=beamFWHM)
Cϕϕkfun{dm,T1,T2}(p::PhaseParms{dm,T1,T2})  = maternk(p.k; ν=νΦ, ρ=ρΦ, σ=σΦ)
parms.ηk[:]       =  ηkfun(parms.k)
parms.Ck[:]       =  Ckfun(parms.k)
parms.C1k[:]      =  C1kfun(parms.k)
parms.C2k[:]      =  C2kfun(parms.k)
parms.CNNk[:]     =  CNNkfun(parms)
parms.Cϕϕk[:]     = Cϕϕkfun(parms)
parms.CZZmk[:]    =  CZZmkfun(parms.Cϕϕk, parms.Ck, parms.ηk, parms.ξk, parms.x, parms.k, parms.deltx, parms.deltk)
parms.CZZmobsk[:] = parms.CZZmk + parms.CNNk 
parms.CZZmobsk[parms.r .< 1]              = Inf 
parms.CZZmobsk[parms.r .>= 0.9*parms.nyq] = Inf 
# Now a High resolution version of PhaseParms for simulation
parmsHR = PhaseParms(d_const, period_const, nside_const)
parmsHR.ηk[:]   =  ηkfun(parmsHR.k)
parmsHR.Ck[:]   =  Ckfun(parmsHR.k)
parmsHR.CNNk[:] =  CNNkfun(parmsHR)
parmsHR.Cϕϕk[:] =  Cϕϕkfun(parmsHR)


# ---- generate the initial simulation and other plotting quantities
zkobs, zk, zx, zx_noϕ, ϕk, ϕx = simNPhaseGRF(parms, parmsHR)
Dθx          = NonstationaryPhase.ϕx_2_Dθx(ϕx, parms) 
Cℓvar        = Cℓvarfun(parms)
Cℓbias       = Cℓbiasfun(parms)
approxCℓbias = approxCℓbiasfun(parms)
estϕk        = estϕkfun(zkobs, parms)


two_sided_cutloci = NonstationaryPhase.cutloci(parms; tmax = 20, deltatmax = 0.01)




# --- get multiple quadratic estimates based on the same ϕx and the average estimate
EestDθx, Eestϕk, estimates_Dθx, avetime2 = get_estimates(100, ϕx, parms, parmsHR) 
println("ave quadratic estimate clock speed =  $avetime2")



#########################################################
#=
Generate plots for the first set of simulations
=#
#########################################################

# --- now evaluate the above `script'
makeplots()
# redo the plot of the estimates to show the two sided cut locus
ax2 = subplot2grid((3,2), (1,0), colspan=2)
	plot(fftshift(parms.x[1]), fftshift(estimates_Dθx[:,1]),"k", label = L"\hat\theta^\prime(x)", linewidth = 1.5, alpha = 0.4)
	for cntr = 2:min(size(estimates_Dθx,2),5)
		plot(fftshift(parms.x[1]), fftshift(estimates_Dθx[:,cntr]),"k", linewidth = 1.5, alpha = 0.4)
	end
	plot(fftshift(parms.x[1]), fftshift(EestDθx), "k--", label = L"average $\hat\theta^\prime(x)$", linewidth = 3)
	plot(fftshift(parms.x[1]), fftshift(Dθx), "b", label = L"\theta^\prime(x)", linewidth = 2.5,alpha = 0.7)
	fill_between(fftshift(parms.x[1]), -two_sided_cutloci, two_sided_cutloci, facecolor="k", alpha=0.2)
	title(L"Quadratic estimate, average quadratic estimate and the simulation truth $\theta^\prime(x)$")
	axis("tight")
	legend(loc="best",fontsize = 10)


# --- save the figures to disk
if savefigures
    fig1[:savefig]("figure3.pdf", dpi=300, bbox_inches="tight", transparent=true)
    plt[:close](fig1)
end 






