#--------------------------------------------------------------------------------------------
#--- Routine to analytically characterize a ZLK resonance and the intensity of SRFs.
#--------------------------------------------------------------------------------------------
#--- Run this file from the terminal with the following syntax:
#---		python kozai_amplitude.py
#--- Settings have to be configured in this file, in the 'USER INPUT' fields.
#--------------------------------------------------------------------------------------------
#--- You could also import the 'main' function elsewhere to catch the returned parameters:
#---		esup, einf, isup, iinf, tk, tg, trs, tts, trp, ttp, r, tt
#--- esup/einf: max/min reached eccentricity in ZLK.
#--- isup/iinf: max/min reached mutual inclination in ZLK [deg].
#--- tk: ZLK characteristic timescale [yr].
#--- tg: relativistic precession characteristic timescale [yr].
#--- trs/trp: stellar/planetary rotational dissipation characteristic timescale [yr].
#--- tts/ttp: stellar/planetary tidal dissipation characteristic timescale [yr].
#--- r: ratio of SRF/ZLK precession characteristic timescales.
#--- tt: decoupling timescale [yr] (experimental, not very accurate).
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

# --- USER INPUT ------------------------------------------

# Example - GJ436b
Ms = 0.445        #Mass of the star [M_Sun]
Mp = 0.0799*317.8 #Mass of the planet [M_Earth]
Mc = 0.1*317.8    #Mass of the perturber [M_Earth]
Rs = 0.449        #Radius of the star [R_Sun]
Rp = 0.374*11.2   #Radius of the planet [R_Earth]
Os = 85.          #Rotation rate of the star [rad/yr]
Op = 55.          #Rotation rate of the planet [rad/yr]
ks = 0.01         #Apsidal constant of the star
kp = 0.25         #Apsidal constant of the planet
Qp = 1e5          #Tidal dissipation parameter of the planet
a0 = 0.35         #Initial separation of the planet [AU]
ac = 5.8          #Separation of the perturber [AU]
e0 = 0.01         #Initial eccentricity of the planet
ec = 0.03         #Eccentricity of the perturber
i0 = 85.          #Initial mutual inclination [deg]
w0 = 0.           #Initial argument of periastron of the planet [deg]


mode = 'All'      #'Transition' or 'SRF' or 'Kozai' or 'All'
verbose = True
debug = False

# ----------------------------------------------------

import numpy as np
from astropy import units as u
from astropy.units import cds
from astropy import constants as const

cds.enable()
custom_mass = u.def_unit('custom_mass', u.AU**3/(const.G*u.yr**2))
orbital_units = [u.mol, u.rad, u.cd, u.AU, u.yr, u.A, custom_mass, u.K]

ME2O = ((1*u.Mearth).decompose(bases=orbital_units)).value
RE2O = ((1*u.Rearth).decompose(bases=orbital_units)).value
MS2O = ((1*u.Msun).decompose(bases=orbital_units)).value
RS2O = ((1*u.Rsun).decompose(bases=orbital_units)).value
G = ((const.G).decompose(bases=orbital_units)).value
C = (const.c.decompose(bases=orbital_units)).value

def solver(f, a, b, num=1000, tol=1e-10, start=1):
	if start > 0:
		x = np.flip(np.linspace(a, b, num=num, endpoint=True))
		y = f(b)
	else:
		x = np.linspace(a, b, num=num, endpoint=True)
		y = f(a)
	s = x[0] - x[1]
	for xx in x:
		z = f(xx)
		if debug: print('f(' + str(xx) + ') = ' + str(z))
		if z is None: continue
		if np.abs(z) < tol: return xx, np.abs(z)
		if y*z < 0.: 
			if start > 0:
				return solver(f, xx, xx + s)
			else:
				return solver(f, xx + s, xx)
	return None, None

def mu1(m0, m1):
	return G*m0*m1/np.sqrt(m0 + m1)

def mu2(m0, m1, m2):
	return G*m2*(m0 + m1)/np.sqrt(m0 + m1 + m2)

def H(e, i, w):
	return (1 + (3./2)*e**2)*(3*np.cos(i)**2 - 1) + (15./2)*e**2*np.sin(i)**2*np.cos(2*w)

def J(m0, m1, m2, a, ac, e, ec, i):
	mu = mu2(m0, m1, m2)/mu1(m0, m1)
	aac = a*(1 - e**2)/ac
	return aac + 2*mu*np.sqrt(aac*(1 - ec**2))*np.cos(i)

def emax(m0, m1, m2, a0, ac, e0, ec, i0, w0):
	J0 = J(m0, m1, m2, a0, ac, e0, ec, i0)
	def to_solve(e):
		i = imin(e0, e, i0, w0)
		if i is None: return None
		JJ = J(m0, m1, m2, a0, ac, e, ec, i)
		return JJ - J0
	estart = 0.
	while to_solve(estart) is None: 
		estart += 1e-4
		if estart > 1.: return None, None
	eend = 1.
	while to_solve(eend) is None: 
		eend -= 1e-4
		if eend < 0.: return None, None
	if debug: 
		print('>>> e_start/e_end:', estart, ';', eend)
		print('>>> f(e_start/e_end):', to_solve(estart), ';', to_solve(eend))
	esup, eps = solver(to_solve, estart, eend)
	return esup, eps

def emax_SRF(m0, m1, m2, a0, ac, ec, r0, r1, O0, O1, k0, k1, i0):
	tk  = tau_k(m0, m1, m2, a0, ac, ec)
	tg  = tau_g(m0, m1, a0, 0.)
	trs = tau_r(m1, m0, r0, O0, k0, a0, 0.)
	tts = tau_t(m1, m0, r0, k0, a0, 0.)
	trp = tau_r(m0, m1, r1, O1, k1, a0, 0.)
	ttp = tau_t(m0, m1, r1, k1, a0, 0.)
	eg  = tk/tg
	ers = tk/trs
	ets = tk/tts
	erp = tk/trp
	etp = tk/ttp
	def to_solve(e):
		f = 1 + 3*e**2 + (3./8)*e**4
		j = np.sqrt(1 - e**2)
		r = eg*(1/j - 1) + (ets + etp)*(f/j**9 - 1)/15 + (ers + erp)*(1/j**3 - 1)/3
		l = 9*e**2*(j**2 - 5*(np.cos(i0)**2)/3)/(8*j**2)
		return r - l
	estart = 1e-4
	eend = 1 - 1e-4
	esup, eps = solver(to_solve, estart, eend, start=1)
	return esup, eps

def emin(m0, m1, m2, a0, ac, e0, ec, i0, w0):
	J0 = J(m0, m1, m2, a0, ac, e0, ec, i0)
	def to_solve(e):
		i = imax(e0, e, i0, w0)
		if i is None: return None
		JJ = J(m0, m1, m2, a0, ac, e, ec, i)
		return JJ - J0
	estart = 0.
	while to_solve(estart) is None: 
		estart += 1e-4
		if estart > 1.: return None, None
	eend = 1.
	while to_solve(eend) is None: 
		eend -= 1e-4
		if eend < 0.: return None, None
	if debug: 
		print('>>> e_start/e_end:', estart, ';', eend)
		print('>>> f(e_start/e_end):', to_solve(estart), ';', to_solve(eend))
	einf, eps = solver(to_solve, estart, eend, start=0)
	return einf, eps

def imax(e0, emin, i0, w0):
	if emin == 1.: return None
	H0 = H(e0, i0, w0)
	s = (H0 + 1 - 6*emin**2)/(3*(1 - emin**2))
	if s < 0: return None
	cos = np.sqrt(s)
	if cos > 1: return None
	return np.arccos(cos)

def imin(e0, emax, i0, w0):
	H0 = H(e0, i0, w0)
	s = (H0 + 1 + 9*emax**2)/(3*(1 + 4*emax**2))
	if s < 0: return None
	cos = np.sqrt(s)
	if cos > 1: return None
	return np.arccos(cos)

def tau_k(m0, m1, m2, a, ac, ec):
	return (16./15)*ac**3*(1 - ec**2)**(3./2)*np.sqrt((m0 + m1)/G)/(a**(3./2)*m2)

def tau_g(m0, m1, a, e):
	P = 2*np.pi*np.sqrt(a**3/(G*(m0 + m1)))
	return (1/(6*np.pi))*(a*(1 - e**2)*C**2/G*(m0 + m1))*P

def tau_r(m0, m1, r1, O1, k1, a, e):
	P = 2*np.pi*np.sqrt(a**3/(G*(m0 + m1)))
	return 4*np.pi*(a/r1)**5*((1 -e**2)**2/k1)*(m1/(m0 + m1))*(1./(P*O1**2))

def tau_t(m0, m1, r1, k1, a, e):
	P = 2*np.pi*np.sqrt(a**3/(G*(m0 + m1)))
	f = 1 + (3./2)*e**2 + (1./8)*e**4
	return (1/(15*np.pi))*(a/r1)**5*((1 - e**2)**5/(k1*f))*(m1/(m0 + m1))*P

def convert_input(m0, m1, m2, a, ac, e, ec, r0, r1, O0, O1, k0, k1, Q1, i, w):
	m0 *= MS2O
	m1 *= ME2O
	m2 *= ME2O
	try:
		r0 *= RS2O
		r1 *= RE2O
		i *= np.pi/180
		w *= np.pi/180
	except TypeError:
		pass
	return m0, m1, m2, a, ac, e, ec, r0, r1, O0, O1, k0, k1, Q1, i, w

def ratio(tk, *args):
	f = np.sum(1/np.asarray(args))
	return tk*f

def tau_trans(m0, m1, m2, a0, ac, ec, r0, r1, O0, O1, k0, k1, Q1, i0):
	'''
	if not log:
		t = (80*a**13*C**2*(-1 + e**2)**6*G*m1*m2)/ \
        	(ac**3*e**2*(1 - ec**2)**1.5*(m0 + m1)*(48*a**4*(-1 + e**2)**4*G**2*m1*(m0 + m1) + 15*C**2*(52 + 50*e**2 + 3*e**4)*G*k*(m0 + m1)*R**5 - \
            16*a**3*C**2*(-1 + e**2)**3*k*O**2*R**5)*((40*cosi**2*np.sqrt((a**13*(-1 + e**2)*G)/((-1 + ec**2)**3*(m0 + m1)))*m2)/ac**3 + \
            (k*m0*(576*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) + 5*e**6*(9*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) - 88*O) - 352*O - 44*e**8*O + \
            60*e**4*(18*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) + 11*O) + 16*e**2*(135*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) + 11*O))*R**5)/((-1 + e**2)**7*m1*Q)))
	else:
		t = (-16*a**5*(-1 + e**2)*(48*a**4*(-1 + e**2)**4*G**2*m1*(m0 + m1) + 15*C**2*(8 + 12*e**2 + e**4)*G*k*(m0 + m1)*R**5 - 8*a**3*C**2*(-1 + e**2)**3*k*O**2*R**5)*
         	np.log(10))/(3.*e**2*(48*a**4*(-1 + e**2)**4*G**2*m1*(m0 + m1) + 15*C**2*(52 + 50*e**2 + 3*e**4)*G*k*(m0 + m1)*R**5 - 
            16*a**3*C**2*(-1 + e**2)**3*k*O**2*R**5)*((40*cosi**2*np.sqrt((a**13*(-1 + e**2)*G)/((-1 + ec**2)**3*(m0 + m1)))*m2)/ac**3 + 
            (k*m0*(576*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) + 5*e**6*(9*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) - 88*O) - 352*O - 44*e**8*O + 
            60*e**4*(18*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) + 11*O) + 16*e**2*(135*np.sqrt(((1 - e**2)*G*(m0 + m1))/a**3) + 11*O))*R**5)/((-1 + e**2)**7*m1*Q)))
	'''
	e, _ = emax_SRF(m0, m1, m2, a0, ac, ec, r0, r1, O0, O1, k0, k1, i0)
	if e is None: return None, None
	a = a0
	R = r1
	O = O1
	k = k1
	Q = Q1
	ad = (-3*k*m0*(-20*a**2*e**10*O + 5*e**8*(5*np.sqrt(a*(1 - e**2)*G*(m0 + m1)) - 64*a**2*O) + 32*e**2*(31*np.sqrt(a*(1 - e**2)*G*(m0 + m1)) - 11*a**2*O) + 
         64*(np.sqrt(-(a*(-1 + e**2)*G*(m0 + m1))) - a**2*O) + 20*e**6*(37*np.sqrt(a*(1 - e**2)*G*(m0 + m1)) + 11*a**2*O) + 
         8*e**4*(255*np.sqrt(a*(1 - e**2)*G*(m0 + m1)) + 67*a**2*O))*R**5)/(16.*a**6*(-1 + e**2)**8*m1*Q)
	t = a/ad
	t /= (1 - e**2)**(1./2)
	return np.abs(t), e

def main(m0, m1, m2, a0, ac, e0, ec, r0=None, r1=None, O0=None, O1=None, k0=None, k1=None, Q1=None, i0=None, w0=None, mode='All', verbose=False):
	
	if mode not in ['Transition', 'SRF', 'Kozai', 'All']: raise SystemExit('Invalid mode.')
	if mode in ['Kozai', 'All'] and (i0 is None or w0 is None): raise SystemExit('Invalid input.')
	if mode in ['Transition', 'SRF', 'All'] and (r0 is None or r1 is None or O0 is None or O1 is None or k0 is None or k1 is None): raise SystemExit('Invalid input.')
	if mode in ['Transition', 'All'] and (Q1 is None or i0 is None): raise SystemExit('Invalid input.')

	m0, m1, m2, a0, ac, e0, ec, r0, r1, O0, O1, k0, k1, Q1, i0, w0 = convert_input(m0, m1, m2, a0, ac, e0, ec, r0, r1, O0, O1, k0, k1, Q1, i0, w0)

	if mode in ['Kozai', 'All']:
		d = True
		esup, epssup = emax(m0, m1, m2, a0, ac, e0, ec, i0, w0)
		if esup is None: 
			if verbose: print('+ No solution found for e_max.')
			d = False
			esup = einf = isup = iinf = None
		if d:
			einf, epsinf = emin(m0, m1, m2, a0, ac, e0, ec, i0, w0)
			if einf is None: 
				if verbose: print('+ No solution found for e_min.')
				esup = einf = isup = iinf = None
				d = False
			if d:
				isup = imax(e0, einf, i0, w0)*180/np.pi
				iinf = imin(e0, esup, i0, w0)*180/np.pi
	else:
		esup = einf = isup = iinf = None
	
	if mode in ['SRF', 'Transition', 'All']:
		tk  = tau_k(m0, m1, m2, a0, ac, ec)
		tg  = tau_g(m0, m1, a0, e0)
		trs = tau_r(m1, m0, r0, O0, k0, a0, e0)
		tts = tau_t(m1, m0, r0, k0, a0, e0)
		trp = tau_r(m0, m1, r1, O1, k1, a0, e0)
		ttp = tau_t(m0, m1, r1, k1, a0, e0)
		r   = ratio(tk, tg, trs, tts, trp, ttp)
	else:
		tk = tg = trs = tts = trp = ttp = r = None

	if mode in ['Transition', 'All']:
		tt, esrf = tau_trans(m0, m1, m2, a0, ac, ec, r0, r1, O0, O1, k0, k1, Q1, i0)
	else:
		tt, esrf = None, None
	
	if verbose:
		if esup is not None:
			print('+ Max/min eccentricity:',       format(esup, '-.3'), ';', format(einf, '-.3'))
			print('+ Min/max mutual inclination:', format(iinf, '-.3'), ';', format(isup, '-.3'), 'deg')
			print('+ Errors:', format(epssup, '.2e'), ';', format(epsinf, '.2e'))
			print('')
		if tk is not None:
			print('+ Kozai timescale:', format(tk, '.2e'), 'yr')
			print('')
			print('+ GR timescale:', format(tg, '.2e'), 'yr')
			print('+ Rotation timescale (planetary/stellar):', format(trp, '.2e'), ';', format(trs, '.2e'), 'yr')
			print('+ Tides timescale    (planetary/stellar):', format(ttp, '.2e'), ';', format(tts, '.2e'), 'yr')
			print('')
			print('+ Ratio of SRF to Kozai precession rates:', format(r, '.2e'))
		if tt is not None:
			print('')
			print('+ Max eccentricity with SRF: {:.3f}'.format(esrf))
			print('+ Transition timescale: {:.2e} yr'.format(tt))
	
	return esup, einf, isup, iinf, tk, tg, trs, tts, trp, ttp, r, tt

if __name__ == '__main__':
	main(Ms, Mp, Mc, a0, ac, e0, ec, Rs, Rp, Os, Op, ks, kp, Qp, i0, w0, mode=mode, verbose=verbose)