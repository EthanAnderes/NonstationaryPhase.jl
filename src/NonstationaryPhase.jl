module NonstationaryPhase

using Distributions

export	PhaseParms,
		fftd,
		ifftd,
		ifftdr,
		invAℓfun,
		estϕkfun,
		unnormalized_estϕkfun,
		Cℓbiasfun,
		approxCℓbiasfun,
		Cℓvarfun,
		gen_tangentMatern,
		simNPhaseGRF,
		maternk,
		σunit_to_σpixl,
		σpixl_to_σunit,
		CZZmkfun,
		CNNkfun,
		squash,
		squash!,
		radial_power

FFTW.set_num_threads(CPU_CORES)

# This source file defines the scaled Fourier transforms used
include("gridfft.jl")

# This source file includes simple implementations, for dimension 1, of the quadratic estimate, Cvar and Cbias 
# which can be used to generate estimates for non-local invariant models. 
# Used in `make_figure_a.jl` and `make_figure_b.jl`
include("nonlocal_invariant_qe.jl")



##########################################################
#=
Definition of the PhaseParms Type.
Holds grid, model and planned FFT parameters for the quadratic estimate.
Allows easy argument passing.
=#
#############################################################

immutable PhaseParms{dm, T1, T2}
	# grid parameters
	period::Float64
	nside::Int64
	deltx::Float64
	deltk::Float64 
	nyq::Float64
	x::Array{Array{Float64,dm},1}
	k::Array{Array{Float64,dm},1}
	r::Array{Float64,dm}
	# parameters necessary for simulation
	ξk::Array{Array{Complex{Float64},dm},1}
	ηk::Array{Array{Float64,dm},1}
	Ck:: Array{Float64,dm}
	CNNk::Array{Float64,dm}
	Cϕϕk::Array{Float64,dm} 
	Eϕx::Array{Float64,dm}
	# parameters necessary for the quad est
	C1k::Array{Array{Complex{Float64},dm},1}
	C2k::Array{Array{Float64,dm},2}
	CZZmk::Array{Float64,dm}
	CZZmobsk::Array{Float64,dm} # used for weights, i.e. masked version of 
	# saved plans for fast fft
	FFT::T1
	IFFT::T2            # this is the unnormalized version
	FFTconst::Float64   # these hold the normalization constants 
	IFFTconst::Float64
end

"""
`PhaseParms(dm, period, nside)` constructor for PhaseParms{dm,T1,T2} type
"""
function PhaseParms(dm, period, nside) 
	dm_nsides = fill(nside,dm)   # [nside,...,nside] <- dm times
	deltx     = period / nside  
	deltk     = 2π / period  
	nyq       = 2π / (2deltx)
	x         = [fill(NaN, dm_nsides...) for i = 1:dm]
	k         = [fill(NaN, dm_nsides...) for i = 1:dm]
	r         = fill(NaN, dm_nsides...) 
	ξk        = [fill(im*NaN, dm_nsides...) for i = 1:dm]
	ηk        = [fill(NaN, dm_nsides...) for i = 1:dm]
	Ck        = fill(NaN, dm_nsides...)
	CNNk      = fill(0.0, dm_nsides...)
	Eϕx       = fill(0.0, dm_nsides...)  # default at zero
	C1k       = [fill(im*NaN, dm_nsides...) for i = 1:dm]
	C2k       = [fill(NaN, dm_nsides...) for i = 1:dm, j = 1:dm]
	CZZmk     = fill(NaN, dm_nsides...)
	CZZmobsk  = fill(NaN, dm_nsides...)
	Cϕϕk      = fill(NaN, dm_nsides...)
	FFT       = plan_fft(rand(Complex{Float64},dm_nsides...); flags = FFTW.PATIENT, timelimit = 10)
	IFFT      = plan_bfft(rand(Complex{Float64},dm_nsides...); flags = FFTW.PATIENT, timelimit = 10)
	parms     = PhaseParms{dm, typeof(FFT), typeof(IFFT)}(period, nside, deltx, deltk, nyq, x, k, r, 
		ξk, ηk, Ck, CNNk, Cϕϕk, Eϕx, 
		C1k, C2k, CZZmk, CZZmobsk, FFT, IFFT, (deltx / √(2π))^dm , (deltk / √(2π))^dm 
		)
	parms.x[:], parms.k[:] = getgrid(parms) 
	parms.r[:]  =  √(sum([abs2(kdim) for kdim in parms.k]))
	parms.ξk[:] = im.*parms.k     # default 
	return parms
end
function getxkside{dm,T1,T2}(g::PhaseParms{dm,T1,T2})
	deltx    = g.period / g.nside   
	deltk    = 2π / g.period   
	xco_side = zeros(g.nside)
	kco_side = zeros(g.nside)
	for j in 0:(g.nside-1)
		xco_side[j+1] = (j < g.nside/2) ? (j*deltx) : (j*deltx - g.period)
		kco_side[j+1] = (j < g.nside/2) ? (j*deltk) : (j*deltk - 2*π*g.nside/g.period)
	end
	xco_side, kco_side
end
function getgrid{T1,T2}(g::PhaseParms{1,T1,T2}) 
	xco_side, kco_side = getxkside(g)
	xco      = Array{Float64,1}[ xco_side ]
	kco      = Array{Float64,1}[ kco_side ]
	return xco, kco
end
function meshgrid(side_x,side_y)
    	nx = length(side_x)
    	ny = length(side_y)
    	xt = repmat(vec(side_x).', ny, 1)
    	yt = repmat(vec(side_y)  , 1 , nx)
    	xt, yt
end
function getgrid{T1,T2}(g::PhaseParms{2,T1,T2}) 
	xco_side, kco_side = getxkside(g)
	kco1, kco2 = meshgrid(kco_side, kco_side)
	xco1, xco2 = meshgrid(xco_side, xco_side)
	kco    = Array{Float64,2}[kco1, kco2]
	xco    = Array{Float64,2}[xco1, xco2]
	return xco, kco
end


import Base.show
function Base.show{dm, T1, T2}(io::IO, parms::PhaseParms{dm, T1, T2})
	for vs in fieldnames(parms)
		(vs != :FFT) && (vs != :IFFT) && println(io, "$vs => $(getfield(parms,vs))")
		println("")
	end
end



##########################################################
#=
Closures for generating ηkfun, Ckfun, C1kfun, C2kfun
=#
#############################################################

"""
Closure which generates a nonstationay phase model in ℝ^d by specifying matern C_k and tildeC_k.
"""
function gen_tangentMatern(; ν = 2.0, ρ = 1.0, σ  = 1.0, tild_ρ = 1.0, tild_ν = 2.5, t_0 = 1.0)
	Ckfun(kco) = maternk(kco; ν=ν, ρ=ρ, σ=σ)
	function ηkfun{dm}(kco::Array{Array{Float64,dm},1})
		d0 = 4ν / ρ / ρ
		dt = 4tild_ν / tild_ρ / tild_ρ
		r2  = zero(kco[1])
		for jj = 1:dm
			r2 += abs2(kco[jj])
		end
		X0  = Distributions.Beta(dm/2, ν)
		Xt  = Distributions.Beta(dm/2, tild_ν)
		F0        = Distributions.cdf(X0, r2 ./ (r2 + d0))
		invFtF0   = (1./Distributions.quantile(Xt, F0) - 1) .^ (-1/2)
		invFtF0 .*= √(dt)
		psiprime = Array{Float64,dm}[ squash( invFtF0.*kco[jj]./√(r2) ) for jj=1:dm]
		return Array{Float64,dm}[ (psiprime[jj] .- kco[jj]) ./ t_0 for jj=1:dm]
	end
	function C1kfun{dm}(kco::Array{Array{Float64,dm},1})
		local ηkval = ηkfun(kco)
		local Ckval = Ckfun(kco)
		local  rtnkimag = Array{Complex{Float64},dm}[ im .* ηkval[j] .* Ckval ./  (2π) ^ (dm/2) for j = 1:dm]
		# Note: the  1/((2π)^(dm/2)) makes C1kfun = fft of C1x
		return rtnkimag::Array{Array{Complex{Float64},dm},1}
	end
	function C2kfun{dm}(kco::Array{Array{Float64,dm},1})
		Ck  = Ckfun(kco)
		C2k = [zero(kco[1]) for i = 1:dm, j = 1:dm]
		for ii = 1:dm, jj = 1:dm
			C2k[ii,jj]   = ηkfun(kco)[ii] .* ηkfun(kco)[jj] .* Ck
			C2k[ii,jj] ./= - 2 * (2π) ^ (dm/2)
		end
		return C2k
	end

	return ηkfun::Function, Ckfun::Function, C1kfun::Function, C2kfun::Function
end



"""
Closure which generates a nonstationay phase model in ℝ^1 by specifying a band limited matern with tangent adjustment.
"""
function gen_bandlimited_tangentMatern(nyq; ν = 2.0, ρ = 1.0, σ  = 1.0, tild_ρ = 1.0, tild_ν = 2.5, t_0 = 1.0)
	function Ckfun{dm}(kco::Array{Array{Float64,dm},1})
		r       = √( sum( Array{Float64,dm}[ abs2(kco[jj]) for jj = 1:dm ] ) )
		tanr    = ((2nyq)/π) .* tan( r .* (π/(2nyq)) ) 
		sec2r   = abs2( sec( r .* (π/(2nyq)) ) )
		tanargs =  Array{Float64,dm}[ squash(tanr .* kco[jj] ./ r) for jj = 1:dm]
		rtnk    = maternk(tanargs; ν=ν, ρ=ρ, σ=σ)
		rtnk  .*= sec2r
		squash!(rtnk)
		return	rtnk
	end
	function ηkfun{dm}(kco::Array{Array{Float64,dm},1})
		d0 = 4ν / ρ / ρ
		dt = 4tild_ν / tild_ρ / tild_ρ
		X0  = Distributions.Beta(dm/2, ν)
		Xt  = Distributions.Beta(dm/2, tild_ν)
		r       = √( sum( Array{Float64,dm}[ abs2(kco[jj]) for jj = 1:dm ] ) )
		tanr    = ((2nyq)/π) .* tan( r .* (π/(2nyq)) ) 
		sec2r   = abs2( sec( r .* (π/(2nyq)) ) )
		F0        = Distributions.cdf(X0, abs2(tanr) ./ (abs2(tanr) + d0))
		invFtF0   = (1./Distributions.quantile(Xt, F0) - 1) .^ (-1/2)
		invFtF0 .*= √(dt)
		invFtF0   = ((2nyq)/π) .* atan( invFtF0 .* (π/(2nyq)) )
		psiprime = Array{Float64,dm}[ squash(invFtF0.*kco[jj]./r) for jj=1:dm]
		return Array{Float64,dm}[ (psiprime[jj] .- kco[jj]) ./ t_0 for jj=1:dm]
	end
	function C1kfun{dm}(kco::Array{Array{Float64,dm},1})
		local ηkval = ηkfun(kco)
		local Ckval = Ckfun(kco)
		local  rtnkimag = Array{Complex{Float64},dm}[ im .* ηkval[j] .* Ckval ./  (2π) ^ (dm/2) for j = 1:dm]
		# Note: the  1/((2π)^(dm/2)) makes C1kfun = fft of C1x
		return rtnkimag::Array{Array{Complex{Float64},dm},1}
	end
	function C2kfun{dm}(kco::Array{Array{Float64,dm},1})
		Ck  = Ckfun(kco)
		C2k = [zero(kco[1]) for i = 1:dm, j = 1:dm]
		for ii = 1:dm, jj = 1:dm
			C2k[ii,jj]   = ηkfun(kco)[ii] .* ηkfun(kco)[jj] .* Ck
			C2k[ii,jj] ./= - 2 * (2π) ^ (dm/2)
		end
		return C2k
	end
	return ηkfun::Function, Ckfun::Function, C1kfun::Function, C2kfun::Function
end




##########################################################
#=
The quadratic estimate, Aℓ, Cℓvar and Cℓbias
=#
#############################################################


"""
Computes the quadratic estimate *with* normalization Aℓ
"""
function estϕkfun{dm,T1,T2}(zk, parms::PhaseParms{dm,T1,T2})
	rtnk   = unnormalized_estϕkfun(zk, parms)
    rtnk ./= invAℓfun(parms)
	squash!(rtnk)
    return rtnk
end



"""
Computes the quadratic estimate before normalization with Aℓ
"""
function unnormalized_estϕkfun{dm,T1,T2}(zk, parms::PhaseParms{dm,T1,T2})
	deltx, deltk   = parms.deltx, parms.deltk
	zkinvCZZ       = similar(zk)
	zkinvCZZ_x_2im = similar(zk)
	for inx in eachindex(zkinvCZZ)
		tmp                 = zk[inx] / parms.CZZmobsk[inx]
		zkinvCZZ[inx]       = isnan(tmp) ? Complex(0.0) : isfinite(tmp) ? tmp : Complex(0.0)
		zkinvCZZ_x_2im[inx] = isnan(tmp) ? Complex(0.0) : isfinite(tmp) ? (2*im*tmp) : Complex(0.0)
	end
	Ax   =  parms.IFFT * (zkinvCZZ) 
	Bpx  = zeros(Complex{Float64}, size(zk))
	rtnk = zeros(Complex{Float64}, size(zk))
	for p = 1:dm
		Bpx[:] = parms.IFFT * (zkinvCZZ_x_2im .* imag(parms.C1k[p])) 
		rtnk  += conj(parms.ξk[p]) .* (parms.FFT * (Ax .* Bpx)) 
	end
	scale!(rtnk, (parms.IFFTconst)^2 * parms.FFTconst)
    return rtnk
end
function unnormalized_estϕkfun{dm,T1,T2}(xk, yk, parms::PhaseParms{dm,T1,T2})
	deltx, deltk = parms.deltx, parms.deltk
	xkinvCZZ     = similar(xk)
	ykinvCZZ     = similar(yk)  
	for inx in eachindex(xkinvCZZ, ykinvCZZ)
		xtmp          = xk[inx] / parms.CZZmobsk[inx]
		ytmp          = yk[inx] / parms.CZZmobsk[inx]
		xkinvCZZ[inx] = isnan(xtmp) ? Complex(0.0) : isfinite(xtmp) ? xtmp : Complex(0.0)
		ykinvCZZ[inx] = isnan(ytmp) ? Complex(0.0) : isfinite(ytmp) ? ytmp : Complex(0.0)
	end
	A1x  =  parms.IFFT * xkinvCZZ 
	A2x  =  parms.IFFT * ykinvCZZ 
	Dpx  = zeros(Complex{Float64}, size(xk))
	Cpx  = zeros(Complex{Float64}, size(xk))
	rtnk = zeros(Complex{Float64}, size(xk))
	for p = 1:dm
		Dpx[:] = parms.IFFT * (parms.C1k[p]  .* ykinvCZZ)
		Cpx[:] = parms.IFFT * (conj(parms.C1k[p]) .* xkinvCZZ)
		rtnk   +=  conj(parms.ξk[p]) .* (parms.FFT * (A1x .* Dpx - A2x .* Cpx))
	end
	scale!(rtnk, (parms.IFFTconst)^2 * parms.FFTconst)
    return rtnk
end



"""
Computes 1/Aℓ when CXXk == CZZmobsk
"""
function invAℓfun{dm,T1,T2}(CXXk, parms::PhaseParms{dm,T1,T2})
	deltx, deltk = parms.deltx, parms.deltk
	CXXinvCZZ = CXXk ./ parms.CZZmobsk ./ parms.CZZmobsk
	squash!(CXXinvCZZ)
	ABCDx = zeros(Complex{Float64}, size(CXXk))
	rtnk  = zeros(Complex{Float64}, size(CXXk))
	for p = 1:dm   # diag terms
		Apqx = ifftd( parms.C1k[p] .* conj(parms.C1k[p]) .* CXXinvCZZ, deltk)
		Bx   = ifftd(                                       CXXinvCZZ, deltk)
		Cpx  = ifftd(                       parms.C1k[p] .* CXXinvCZZ, deltk)
		Dpx  = ifftd(                 conj(parms.C1k[p]) .* CXXinvCZZ, deltk)
		for ix in eachindex(ABCDx)
			ABCDx[ix] = 2 * Apqx[ix] * Bx[ix] - Cpx[ix] * Cpx[ix] - Dpx[ix] * Dpx[ix]
		end
		rtnk += abs2(parms.ξk[p]) .* fftd(ABCDx, deltx)
	end
	for p = 1:dm, q = (p+1):(dm)   # off diag terms
		Apqx = ifftd( parms.C1k[p] .* conj(parms.C1k[q]) .* CXXinvCZZ, deltk)
		Bx   = ifftd(                                       CXXinvCZZ, deltk)
		Cpx  = ifftd(                       parms.C1k[p] .* CXXinvCZZ, deltk)
		Cqx  = ifftd(                       parms.C1k[q] .* CXXinvCZZ, deltk)
		Dpx  = ifftd(                 conj(parms.C1k[p]) .* CXXinvCZZ, deltk)
		Dqx  = ifftd(                 conj(parms.C1k[q]) .* CXXinvCZZ, deltk)
		for ix in eachindex(ABCDx)
			ABCDx[ix] = 2 * Apqx[ix] * Bx[ix] - Cpx[ix] * Cqx[ix] - Dpx[ix] * Dqx[ix]
		end
		rtnk += 2 * parms.ξk[p] .* conj(parms.ξk[q]) .* fftd(ABCDx, deltx)
	end
	return rtnk::Array{Complex{Float64},dm}
end
function invAℓfun{dm,T1,T2}(parms::PhaseParms{dm,T1,T2})
	CXXk = parms.CZZmk + parms.CNNk
	return invAℓfun(CXXk, parms)
end



"""
Computes Cℓvar, i.e. the variance spectral density of the quadratic estimate
"""
function Cℓvarfun{dm,T1,T2}(CXXk, parms::PhaseParms{dm,T1,T2})
	invAℓ = invAℓfun(parms)
	rtnk  = invAℓfun(CXXk, parms) ./ invAℓ ./ invAℓ
	squash!(rtnk)
	scale!(rtnk, 2 * (2π) ^ (-dm/2) )
	return real(rtnk)
end
function Cℓvarfun{dm,T1,T2}(parms::PhaseParms{dm,T1,T2})
	invAℓ = invAℓfun(parms)
	rtnk  = 1.0 ./ invAℓ
	squash!(rtnk)
	scale!(rtnk,  2 * (2π) ^ (-dm/2) )
	return real(rtnk)
end





"""
Cℓbiasfun. Function for computing the second order bias in the quadratic estimate.
"""
function Cℓbiasfun{dm,T1,T2}(parms::PhaseParms{dm,T1,T2})
	m1s   = ones(parms.Cϕϕk)
	boxbool = trues(size(parms.Cϕϕk))
	find_boxbool = find(boxbool)
    rtnkdiag  = zeros(parms.x[1])
    for p = 1:dm, q = 1:dm, p′ = 1:dm, q′ = 1:dm
    	ϕC2kℓ_pq   =  unnormalized_estϕkfun(parms.C2k[p,q],   m1s, parms) 
    	ϕC2kℓ_p′q′ =  unnormalized_estϕkfun(parms.C2k[p′,q′], m1s, parms) 
    	biasupdate!(p,q,p′,q′, parms, 
    		ϕC2kℓ_pq,
    		ϕC2kℓ_p′q′, 
    		m1s, 
    		find_boxbool, 
    		rtnkdiag
    		)
    end
    constant = 4 *  (parms.deltk / 2π) ^ dm 
    rtnk = constant .* rtnkdiag ./ abs2(invAℓfun(parms))
    squash!(rtnk)
    return rtnk
end
function biasupdate!{dm,T1,T2}(p,q,p′,q′, parms::PhaseParms{dm,T1,T2}, ϕC2kℓ_pq, ϕC2kℓ_p′q′, m1s, find_boxbool, storage::Array{Float64,dm})
    	for ωinx in find_boxbool
    		ω = Float64[parms.k[jj][ωinx] for jj = 1:dm]
        	C2kminusω_pq      = shiftfk_by_ω(parms.C2k[p,q],   -ω, parms)
        	C2kminusω_p′q′    = shiftfk_by_ω(parms.C2k[p′,q′], -ω, parms)
    		ϕC2kℓminusω_pq    =  unnormalized_estϕkfun(C2kminusω_pq,  m1s, parms) 
    		ϕC2kℓminusω_p′q′  =  unnormalized_estϕkfun(C2kminusω_p′q′, m1s, parms) 
        	Cθθℓ_ωqp′   =  shiftfk_by_ω( parms.ξk[q] .* conj(parms.ξk[p′]) .* parms.Cϕϕk, -ω, parms) 
        	Cθθℓ_ωqq′   =  shiftfk_by_ω( parms.ξk[q] .* conj(parms.ξk[q′]) .* parms.Cϕϕk, -ω, parms) 
        	Cθθωpq′     = parms.ξk[p][ωinx] * conj(parms.ξk[q′][ωinx]) * parms.Cϕϕk[ωinx]
        	Cθθωpp′     = parms.ξk[p][ωinx] * conj(parms.ξk[p′][ωinx]) * parms.Cϕϕk[ωinx]
        	myupdateforbias!(
        		conj(ϕC2kℓ_p′q′ - ϕC2kℓminusω_p′q′), 
        		ϕC2kℓ_pq - ϕC2kℓminusω_pq, 
        		Cθθℓ_ωqq′,
				Cθθℓ_ωqp′,
        		Cθθωpp′,
        		Cθθωpq′, 
        		storage
        		)
    	end
end
function myupdateforbias!{dm}(𝓒pq, 𝓒p′q′, Cθℓωqq′, Cθℓωqp′, Cθωpp′::Number, Cθωpq′::Number, storage::Array{Float64,dm})
	@inbounds for ind in eachindex(𝓒pq, 𝓒p′q′, Cθℓωqq′, Cθℓωqp′, storage)
		storage[ind] += real(𝓒pq[ind]  * 𝓒p′q′[ind]  * (Cθωpp′ * Cθℓωqq′[ind] +  Cθωpq′ * Cθℓωqp′[ind]))
	end
end
function shiftfk_by_ω{dm,T1,T2}(fk, ω, parms::PhaseParms{dm,T1,T2})
	fx    = parms.IFFT * fk
	ωdx  = sum([ω[jj] .* parms.x[jj] for jj = 1:dm ])
	rtnk  = parms.FFT * ( exp(-im .* ωdx) .* fx )
	scale!(rtnk, parms.IFFTconst * parms.FFTconst)
 	return rtnk
end





"""
approxCℓbiasfun
"""
function approxCℓbiasfun{dm,T1,T2}(parms::PhaseParms{dm,T1,T2})
	m1s    = ones(parms.Cϕϕk)
	∂kC2k  = make∂kC2k(parms)
	∂∂kC2k = make∂∂kC2k(parms)
	Cθθk   = makeCθθk(parms)
	rtnk   = zeros(Complex{Float64}, size(parms.k[1]))
    for p = 1:dm, q = 1:dm, p′ = 1:dm, q′ = 1:dm, ∂1 = 1:dm, ∂2 = 1:dm
    	htmp1 =      unnormalized_estϕkfun(∂kC2k[p,q,∂1]  , m1s, parms)
    	htmp2 = conj(unnormalized_estϕkfun(∂kC2k[p′,q′,∂2], m1s, parms))
    	θtmp1 = convtrans(parms.k[∂1].*parms.k[∂2].*Cθθk[p,p′], Cθθk[q,q′], parms)
    	θtmp2 = convtrans(parms.k[∂1].*parms.k[∂2].*Cθθk[p,q′], Cθθk[q,p′], parms)
    	rtnk += htmp1 .* htmp2 .* (θtmp1 + θtmp2)
    end
    for p = 1:dm, q = 1:dm, p′ = 1:dm, q′ = 1:dm, ∂1 = 1:dm, ∂2 = 1:dm, ∂3 = 1:dm
    	htmp1 =      unnormalized_estϕkfun(     ∂kC2k[p,q,∂1],       m1s, parms)
    	htmp2 = conj(unnormalized_estϕkfun(-0.5*∂∂kC2k[p′,q′,∂2,∂3], m1s, parms))
    	θtmp1 = convtrans(parms.k[∂1].*parms.k[∂2].*parms.k[∂3].*Cθθk[p,p′], Cθθk[q,q′], parms)
    	θtmp2 = convtrans(parms.k[∂1].*parms.k[∂2].*parms.k[∂3].*Cθθk[p,q′], Cθθk[q,p′], parms)
    	rtnk += htmp1 .* htmp2 .* (θtmp1 + θtmp2)
    end
    for p = 1:dm, q = 1:dm, p′ = 1:dm, q′ = 1:dm, ∂1 = 1:dm, ∂2 = 1:dm, ∂3 = 1:dm
    	htmp1 =      unnormalized_estϕkfun(-0.5*∂∂kC2k[p,q,∂1,∂2], m1s, parms)
    	htmp2 = conj(unnormalized_estϕkfun(     ∂kC2k[p′,q′,∂3]  , m1s, parms))
    	θtmp1 = convtrans(parms.k[∂1].*parms.k[∂2].*parms.k[∂3].*Cθθk[p,p′], Cθθk[q,q′], parms)
    	θtmp2 = convtrans(parms.k[∂1].*parms.k[∂2].*parms.k[∂3].*Cθθk[p,q′], Cθθk[q,p′], parms)
    	rtnk += htmp1 .* htmp2 .* (θtmp1 + θtmp2)
    end
    for p = 1:dm, q = 1:dm, p′ = 1:dm, q′ = 1:dm, ∂1 = 1:dm, ∂2 = 1:dm, ∂3 = 1:dm, ∂4 = 1:dm
    	htmp1 =      unnormalized_estϕkfun(-0.5*∂∂kC2k[p,q,∂1,∂2]   , m1s, parms)
    	htmp2 = conj(unnormalized_estϕkfun(-0.5*∂∂kC2k[p′,q′,∂3,∂4] , m1s, parms))
    	θtmp1 = convtrans(parms.k[∂1].*parms.k[∂2].*parms.k[∂3].*parms.k[∂4].*Cθθk[p,p′], Cθθk[q,q′], parms)
    	θtmp2 = convtrans(parms.k[∂1].*parms.k[∂2].*parms.k[∂3].*parms.k[∂4].*Cθθk[p,q′], Cθθk[q,p′], parms)
    	rtnk += htmp1 .* htmp2 .* (θtmp1 + θtmp2)
    end
    rtnk .*= 4 * ((2π) ^ (-dm/2)) 
    rtnk ./= abs2(invAℓfun(parms))
    squash!(rtnk)
    return real(rtnk)
end
# --- these are helper functions for approxCℓbias
function make∂kC2k{dm,T1,T2}(parms::PhaseParms{dm,T1,T2})
	rtnk = [zeros(Complex{Float64},size(parms.k[1])) for p = 1:dm, q = 1:dm, deriv = 1:dm]
	for p = 1:dm, q = 1:dm, deriv = 1:dm
		C2xpq  = ifftd(parms.C2k[p,q], parms.deltk)
		rtnk[p,q,deriv][:] =  fftd(-im .* parms.x[deriv] .* C2xpq, parms.deltx) 
	end
	return rtnk
end
function make∂∂kC2k{dm,T1,T2}(parms::PhaseParms{dm,T1,T2})
	rtnk = [zeros(Complex{Float64},size(parms.k[1])) for p = 1:dm, q = 1:dm, ∂1 = 1:dm, ∂2 = 1:dm]
	for p = 1:dm, q = 1:dm, ∂1 = 1:dm, ∂2 = 1:dm
		C2xpq  = ifftd(parms.C2k[p,q], parms.deltk)
		rtnk[p,q,∂1,∂2][:] =  fftd( - parms.x[∂1] .* parms.x[∂2] .* C2xpq, parms.deltx) 
	end
	return rtnk
end
function makeCθθk{dm,T1,T2}(parms::PhaseParms{dm,T1,T2})
	rtnk = [zeros(Complex{Float64},size(parms.k[1])) for p = 1:dm, q = 1:dm]
	for p = 1:dm, q = 1:dm
		rtnk[p,q][:] =  parms.ξk[p] .* conj(parms.ξk[q]) .* parms.Cϕϕk
	end
	return rtnk
end
function convtrans{dm,T1,T2}(Ak, Bk, parms::PhaseParms{dm,T1,T2})
	rtnk   = parms.FFT * ( (parms.IFFT * Ak) .* (parms.IFFT * Bk) ) 
 	rtnk .*= parms.FFTconst .* parms.IFFTconst ^ 2
 	return rtnk
end



##########################################################
#=
The marginal spectral density, CZZmkfun
=#
#############################################################


"""
Computes the marginal spectral density
"""
function CZZmkfun{dm}(Cϕϕk::Array{Float64,dm}, Ck, ηk, ξk, x, k, deltx, deltk)
	abs2k = sum([abs2(kdim) for kdim in k])
	index_xeq0 = findmin(abs2k)[2]
	Σθ = [zero(x[1]) for i = 1:dm, j = 1:dm]
	for ii = 1:dm, jj = 1:dm
		Cθix_θjx   = ifftdr( ξk[ii] .* conj(ξk[jj]) .* Cϕϕk , deltk) 
		Cθix_θjx .*= (2π) ^ (-dm/2)
		Σθx_ij     = Cθix_θjx[index_xeq0] .- Cθix_θjx
		Σθx_ij   .*= 2
		Σθ[ii,jj]  = copy(Σθx_ij)
	end
	imx    = im .* x
	CZmx   = zeros(Complex{Float64}, size(Cϕϕk))
	tmp    = zeros(Complex{Float64}, size(Cϕϕk))
	tmp1   = zeros(Float64, size(Cϕϕk))
	for yi in eachindex(CZmx)
		tmp[:] = Complex(0.0)
		tmp1[:] = 0.0
		for dims1 = 1:dm
			BLAS.axpy!(imx[dims1][yi], k[dims1], tmp)
			for dims2 = 1:dm
				myscaleadd!(-0.5*Σθ[dims1,dims2][yi], ηk[dims1], ηk[dims2], tmp1)
			end
			BLAS.axpy!(1.0, tmp1, tmp)
		end
		CZmx[yi] = fastsumXexpY(Ck,tmp)
	end
	scale!((deltk ^ dm) * ((2π) ^ (-dm)), CZmx)
	CZZmk   = fftd(CZmx, deltx) 
	scale!((2π) ^ (dm/2) , CZZmk)
	return abs(real(CZZmk))
end
function myscaleadd!(number, mat1, mat2, storage)
	@inbounds for ind in eachindex(mat1, mat2, storage)
		storage[ind] = storage[ind] + number * mat1[ind] * mat2[ind]
	end
end


##########################################################
#=
Simulation 
=#
#############################################################
"""
# Simulate nonstationary phase model.
"""
function simNPhaseGRF{dm,T1,T2}(ϕx, parms::PhaseParms{dm,T1,T2}, parmsHR::PhaseParms{dm,T1,T2})
	ϕk     = fftd(ϕx, parms.deltx)
	boldϕx = Array{Float64,dm}[ ifftdr(parms.ξk[j] .* ϕk, parms.deltk) for j = 1:dm ]
	imboldϕx = im .* boldϕx
	imx      = im .* parms.x
	hzx_noϕ  = grfsimx(parmsHR.Ck, parmsHR.deltx, parmsHR.deltk)
	hzk_noϕ  = fftd(hzx_noϕ, parmsHR.deltx)
	zx  = zeros(Complex{Float64}, size(ϕx))
	tmp = zeros(Complex{Float64}, size(hzk_noϕ))
	for yi in eachindex(zx)
		tmp[:] = Complex(0.0)
		for dims = 1:dm   # can you loop unroll this? would it help?
			BLAS.axpy!(imboldϕx[dims][yi], parmsHR.ηk[dims], tmp)
			BLAS.axpy!(imx[dims][yi], parmsHR.k[dims], tmp)
		end
		zx[yi] = fastsumXexpY(hzk_noϕ, tmp)
	end
	zx   .*= (parmsHR.deltk ^ dm) / (2π) ^ (dm/2)
	zk     = fftd(real(zx), parms.deltx)
	zx_noϕ = downsample(hzx_noϕ, Int64(parmsHR.nside/parms.nside))
	zkobs  = zk + fftd(grfsimx(parms.CNNk, parms.deltx, parms.deltk), parms.deltx)
	return zkobs, zk, real(zx), zx_noϕ, ϕk, ϕx
end
function simNPhaseGRF{dm,T1,T2}(parms::PhaseParms{dm,T1,T2}, parmsHR::PhaseParms{dm,T1,T2})
	ϕx   = grfsimx(parms.Cϕϕk, parms.deltx, parms.deltk)
	BLAS.axpy!(1.0, parms.Eϕx, ϕx)
	zkobs, zk, zx, zx_noϕ, ϕk, ϕx = simNPhaseGRF(ϕx, parms, parmsHR)
	return zkobs, zk, zx, zx_noϕ, ϕk, ϕx
end
function fastsumXexpY(hzk, tmp) # used in simNPhaseGRF
	rtnk = Complex(0.0)
	@inbounds for ind in eachindex(hzk,tmp)
		rtnk += hzk[ind] * exp(tmp[ind])
	end
	rtnk
end
downsample{T}(mat::Array{T,1}, hfcr::Int64) =  mat[1:hfcr:end]
downsample{T}(mat::Array{T,2}, hfcr::Int64) =  mat[1:hfcr:end, 1:hfcr:end]
downsample{T}(mat::Array{T,3}, hfcr::Int64) =  mat[1:hfcr:end, 1:hfcr:end, 1:hfcr:end]
# Simulate a mean zero Gaussian random field in the pixel domain given a spectral density.
function grfsimx{T,dm}(Ckvec::Array{T,dm}, deltx, deltk)
	nsz = size(Ckvec)
	dx  = deltx ^ dm
	zzk = √(Ckvec) .* fftd(randn(nsz)./√(dx), deltx)
	return ifftdr(zzk, deltk)::Array{Float64,dm}
end



#  converting from pixel noise std to noise per-unit pixel
σunit_to_σpixl(σunit, deltx, dm) = σunit / √(deltx ^ dm)
σpixl_to_σunit(σpixl, deltx, dm) = σpixl * √(deltx ^ dm)
function CNNkfun{dm}(k::Array{Array{Float64,dm},1}, deltx; σpixl=0.0, beamFWHM=0.0)
	local absk2  = mapreduce(abs2, +, k)::Array{Float64,dm}
	local beamSQ = exp(- (beamFWHM ^ 2) * (absk2 .^ 2) ./ (8 * log(2)) )
	return ones(size(k[1])) .* σpixl_to_σunit(σpixl, deltx, dm) .^ 2 ./ beamSQ
end


function maternk{dm}(kco::Array{Array{Float64,dm},1}; ν=1.1, ρ=1.0, σ=1.0)
    d1 = 4ν / ρ / ρ
    cu = ((2π) ^ dm) * (σ ^ 2) * gamma(ν + dm/2) * ((4ν) ^ ν)
	ed = (π ^ (dm/2)) * gamma(ν) * (ρ ^ (2ν))
	# note...the extra ((2π) ^ (dm)) is so the integral equals σ^2 = ∫ C_k dk/((2π) ^ (dm))
	absk2  = mapreduce(abs2, +, kco)::Array{Float64,dm}
	rtn = (cu / ed) ./ ((d1 +  absk2) .^ (ν + dm/2))
    return rtn
end




##########################################################
#=
Cut locus
=#
#############################################################
"""
# two sided cut locus code
"""
function cutloci(parms; tmax = 20, deltatmax = 0.01)
	cutlociPos = Inf
	for t = 0:deltatmax:tmax   
		if isa_cutloci(t, parms)
			cutlociPos = t
			break
		end
	end
	cutlociNeg = -Inf
	for t = 0:deltatmax:tmax
		if isa_cutloci(-t, parms)
			cutlociNeg = -t
			break
		end
	end
	return min(-cutlociNeg, cutlociPos) 
end
function isa_cutloci(t, parms)
	imk   = fftshift(parms.k[1] + t*parms.ηk[1])
	return any(diff(imk) .<= 0) ? true : false
end

# this is currently un-used in but is instructive to compare with cutloci(parms)
function alternative_cutloci(parms; tmax = 20, deltatmax = 0.01)
	Ckprob   = fftshift(parms.Ck) 
	Ckprob ./= sum(Ckprob)
	cutloci = Inf
	for t = 0:deltatmax:tmax 
		Finv = quantile(fftshift(parms.k[1]), Ckprob)
		Fptinv = quantile(fftshift(parms.k[1] + t*parms.ηk[1]), Ckprob)
		Fntinv = quantile(fftshift(parms.k[1] - t*parms.ηk[1]), Ckprob)
		testratio = (Finv - Fptinv) ./ (Fntinv -  Fptinv)
		max_test = maximum(testratio)
		min_test = minimum(testratio)
		if (abs(max_test-min_test) > 1e-10) | (min_test < 0.) | (max_test > 1.)
			cutloci = t
			break
		end
	end
	return cutloci
end





##########################################################
#=
Miscellaneous functions
=#
#############################################################
function radial_power{dm,T1,T2}(fk, smooth::Number, parms::PhaseParms{dm,T1,T2})
	rtnk = Float64[]
	dk = parms.deltk
	kbins = collect((smooth*dk):(smooth*dk):(parms.nyq))
	for wavenumber in kbins
		indx = (wavenumber-smooth*dk) .< parms.r .<= (wavenumber+smooth*dk)
		push!(rtnk, sum(fk[indx]) / sum(indx))
	end
	return kbins, rtnk 
end



squash{T<:Number}(x::T)         = isnan(x) ? zero(T) : isfinite(x) ? x : zero(T)
squash{T<:AbstractArray}(x::T)  = map(squash, x)::T
squash!{T<:AbstractArray}(x::T) = map!(squash, x)::T


function isgoodfreq{dm}(prop_of_nyq, deltx, kco::Array{Array{Float64,dm},1})  
	# for generating a boolean mask
	magk = √(mapreduce(abs2, +, kco))::Array{Float64,dm}
	isgood = (magk .<= (prop_of_nyq * π / deltx)) & (magk .> 0.0)
	return isgood
end

function tent(x::Real, lend=π/2, uend=3π/2)
	# looks like this _/\_ with derivative ±1.`
	midd = lend + (uend - lend) / 2
	rtn  = 	(x ≤ lend) ? 0.0 :
			(x ≤ midd) ? ( x - lend) :
			(x ≤ uend) ? (-x + uend) : 0.0
	return rtn
end
function tent{T<:Real}(x::Array{T,1}, lend=π/2, uend=3π/2)
	return map(xr->tent(xr, lend, uend), x)
end


function Dθx_2_ϕx{dm,T1,T2}(Dθx, parms::PhaseParms{dm,T1,T2}) 
	Dθk = fftd(Dθx, parms.deltx)
	θk  = squash(Dθk ./ (im .* parms.k[1]))
	ϕk  = squash(θk ./ parms.ξk[1])
	ϕx = ifftdr(ϕk, parms.deltk)
	return ϕx
end
function ϕx_2_Dθx{dm,T1,T2}(ϕx, parms::PhaseParms{dm,T1,T2}) 
	ϕk = fftd(ϕx, parms.deltx)
	θk  = ϕk .* parms.ξk[1]
	Dθk  = θk .* (im .* parms.k[1])
	Dθx = ifftdr(Dθk, parms.deltk)
	return Dθx
end
function ϕk_2_Dθx{dm,T1,T2}(ϕk, parms::PhaseParms{dm,T1,T2}) 
	θk  = ϕk .* parms.ξk[1]
	Dθk  = θk .* (im .* parms.k[1])
	Dθx = ifftdr(Dθk, parms.deltk)
	return Dθx
end
function ϕk_2_θx{dm,T1,T2}(ϕk, parms::PhaseParms{dm,T1,T2}) 
	θk  = ϕk .* parms.ξk[1]
	θx = ifftdr(θk, parms.deltk)
	return θx
end



end # module
