from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from scipy.optimize import minimize

k_dm = const.e.gauss**2/(2*np.pi*const.m_e*const.c)

def abs2(x):
    return x.real**2 + x.imag**2

def dphi(dm, dphi_ref, ppsr, freqs, freq_ref):
    dphi = dphi_ref + (k_dm * (dm*u.pc/u.cm**3) / ppsr * (freqs**(-2) - freq_ref**(-2))).to(u.dimensionless_unscaled)
    return dphi

def Cxx_chan(data_f):
    return np.sum(abs2(data_f)[1:], axis=0)

def Cxy_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs, freq_ref):
    dm, dphi_ref = p
    num = np.real( data_f * tmplt_f.conj() * np.exp(2j*np.pi*spin_freqs[:,np.newaxis]*dphi(dm, dphi_ref, ppsr, freqs, freq_ref)) )
    return np.sum(num[1:], axis=0)

def amps_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs, freq_ref):
    return Cxy_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs, freq_ref)/Cxx_chan(tmplt_f)



def chi_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, nph, freq_ref):
    chisq = Cxx_chan(data_f) / data_var - Cxy_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs, freq_ref)**2 / Cxx_chan(tmplt_f) / data_var
    return chisq

def chi_all(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, nph, freq_ref):
    chisq_chan = chi_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, nph, freq_ref)
    return np.sum(chisq_chan)



def Cxyd1_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs, freq_ref):
    dm, dphi_ref = p
    num = np.real( data_f * tmplt_f.conj() * 2j*np.pi*spin_freqs[:,np.newaxis] * np.exp(2j*np.pi*spin_freqs[:,np.newaxis]*dphi(dm, dphi_ref, ppsr, freqs, freq_ref)) )
    return np.sum(num[1:], axis=0)

def Cxyd2_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs, freq_ref):
    dm, dphi_ref = p
    num = np.real( data_f * tmplt_f.conj() * - (2*np.pi*spin_freqs[:,np.newaxis])**2 * np.exp(2j*np.pi*spin_freqs[:,np.newaxis]*dphi(dm, dphi_ref, ppsr, freqs, freq_ref)) )
    return np.sum(num[1:], axis=0)

def w_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, freq_ref):
    params = [p, data_f, tmplt_f, ppsr, spin_freqs, freqs, freq_ref]
    return (Cxyd2_chan(*params) * Cxy_chan(*params) + Cxyd1_chan(*params)**2) / Cxx_chan(tmplt_f) / data_var



def calc_errors(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, freq_ref):
    wn = w_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, freq_ref)
    xn = k_dm/ppsr * (freqs**(-2) - freq_ref**(-2))
    d2chisqdphi_ref2 = -2 * np.sum( wn )
    d2chisqdm2 = -2 * np.sum( xn**2 * wn )
    d2chisqdan2 = 2 * Cxx_chan(tmplt_f)/data_var
    dm_err = 1/np.sqrt(d2chisqdm2)
    dphi_ref_err = 1/np.sqrt(d2chisqdphi_ref2)
    an_err = 1/np.sqrt(d2chisqdan2)

    return dm_err.to(u.pc/u.cm**3).value, dphi_ref_err, an_err

def chi_check(dt,z_f,z_var,pp_f,freqs,ngates):
    a = np.sum( (z_f*pp_f.conj()*np.exp(1j*2*np.pi*freqs*dt) + z_f.conj()*pp_f*np.exp(-1j*2*np.pi*freqs*dt))[1:] ) / np.sum( 2 * np.abs(pp_f[1:])**2 )
    num = (z_f - (a * pp_f * np.exp(-1j*2*np.pi*freqs*dt)))[1:] # consider k>0
    chisq = np.sum(np.abs(num)**2) / (z_var * ngates/2)
    return chisq

def template_match(template, z, off_gates = None, smart_guess = True, xguess=0., method='Nelder-Mead'):
    """
    Template matching code from Pulsar Timing and Relativistic Gravity, Taylor 1992, appendix A
    Returns delta_t, error in delta_t, scale, and baseline shift needed to match the profile with the provided template.
    Fang Xi Lin, Rob Main, 2019
    Arguments
    ----------
    template : np_array
        Template profile
    z :
        Profile to match the template to
    off_gates :
        indices of the off pulse in z. If not provided, assume gatess below the median are off pulse. Used to estimate errors.
    smart_guess :
        By default, we cross correlate template profile and profile to be matched, and use the highest bin as the initial guess for dt.
    xguess :
        Only used if smart_guess is set to False. Initial guess `dt` for the minimization.
    method :
        Method to use in scipy.optimize.minimize in the chisquare minimization. Nelder-Mead typically performs well, given a reasonable initial guess.
        However, it does tend to converge to local minima if the inital guess is far from the true values.
    """
    ngates = template.size

    assert z.size == ngates, "Template profile and profile to fit are not of the same size. Perhaps rebin one of them?"
    ## TODO: make it work despite the different lengths

    t = np.linspace(0,1,ngates,endpoint=False)
    t += (t[1]-t[0])/2
    ppdt = t[1]-t[0]

    freqs = np.fft.rfftfreq(ngates,ppdt)

    template_f = np.fft.rfft(template)
    z_f = np.fft.rfft(z)

    # if off_gates is provided, use that to compute variance, otherwise, use gates below the median.
    if off_gates is not None:
        z_var = np.sum(np.var(z[off_gates])) # variance in the off pulse, for chisquare purposes
    else:
        z_var = np.sum(np.var(z[z<np.median(z)]))

    if smart_guess == True:
    # cross-correlate profiles, take peak bin to get starting guess
        profcorr = np.fft.irfft( (z_f * template_f.conj())[1:] )
        x = np.fft.fftfreq(len(template))
        xguess = x[np.argmax(np.abs(profcorr))]

    minchisq = minimize(chi_check, x0=xguess, args=(z_f,z_var,template_f,freqs,ngates),method=method)
    if minchisq.success != True:
        print('Chi square minimization failed to converge. !!BEWARE!!')

    dt = minchisq.x

    # error term is eq. A10 from the paper, massaged a bit.
    a = np.sum( (z_f*template_f.conj()*np.exp(1j*2*np.pi*freqs*dt) + z_f.conj()*template_f*np.exp(-1j*2*np.pi*freqs*dt))[1:] ) / np.sum( 2 * np.abs(template_f[1:])**2 )

    dterr = np.sqrt( (z_var*ngates/2)/a / np.sum( ((2*np.pi*freqs)**2*(z_f*template_f.conj()*np.exp(1j*2*np.pi*freqs*dt)+z_f.conj()*template_f*np.exp(-1j*2*np.pi*freqs*dt)))[1:] ).real )

    aerr = np.sqrt( (z_var * ngates/2) / np.sum( 2 * template_f * template_f.conj() ) )

    b = (z_f[0] - a * template_f[0]).real/ngates

    dt = np.real(dt)
    dterr = np.real(dterr)
    a = np.real(a)
    aerr = np.real(aerr)
    b = np.real(b)

    return dt, dterr, a, aerr, b

def shift(z, dt):
    """
    Sub-bin shifting
    returns z shifted by dt in units of cycle (ie., dt=1 returns the same z).
    Arguments
    ---------
    z :
        Profile to shift
    dt :
        Phase to shift
    """

    ngates = z.size
    freqs = np.fft.rfftfreq(ngates,1./ngates)
    return np.real(np.fft.irfft(np.exp(-1j*2*np.pi*freqs*dt)*np.fft.rfft(z)))

def rebin(arr,nbin):
    newshape = arr.shape[:-1]+(-1,nbin,)
    return arr.reshape(*newshape).mean(-1)

def disperse(pulse, freqs, dm, fref, ppsr):
    dts = k_dm*dm/ppsr * (freqs**(-2) - fref**(-2))
    pulse_shifted = pulse.copy()
    for i in range(pulse.shape[-1]):
        pulse_shifted[:,i] = shift(pulse[:,i],dts[i])
    return pulse_shifted

def shift2(z, dt):
    """
    Sub-bin shifting
    returns z shifted by dt in units of cycle (ie., dt=1 returns the same z).
    Arguments
    ---------
    z :
        Profile to shift
    dt :
        Phase to shift
    """

    ngates = z.shape[0]
    freqs = np.fft.rfftfreq(ngates,1./ngates)
    return np.real(np.fft.irfft(np.exp(-1j*2*np.pi*freqs[:,np.newaxis]*dt)*np.fft.rfft(z,axis=0),axis=0))

def fit_dm(data, freqs, nchunk, template = None, ppsr = None, shift_data = False, inf_ref = False):

    """
    Fits DM with n frequency chunks

    Arguments
    ---------

    data :
        data cube to fit, of shape (time_bins, phase_bins, frequency_bins)

    freqs :
        Frequency array with dimensions of frequency.

    nchunk :
        Number of frequency chunks to fit. data is rebinned in frequency to nchunk before fitting.

    template :
        template to fit.

    ppsr :
        pulsar period with dimensions of time.

    shift_data :
        whether to shift input data to align with the template

    inf_ref :
        whether to use infinite frequency as reference

    Output
    ------

    dm :
        DM in pc/cm^3

    dm_err :
        Error in DM in pc/cm^3

    data :
        shifted data if data_shift is True, otherwise original data.
    """

    nt, nph, nf = data.shape[0], data.shape[1], data.shape[2]

    ph = np.linspace(0,1,nph,endpoint=False)
    ph += (ph[1]-ph[0])/2
    dph = ph[1]-ph[0]
    spin_freqs = np.fft.rfftfreq(nph, dph)

    assert freqs.size == nf
    assert nf%nchunk == 0

    if ppsr == None:
        print('Please set pulsar period!')
        return

    if np.shape(template) == ():
        tmplt = data[-250:,...].mean(0)
    else:
        tmplt = template
    if nchunk != nf:
        freqs = rebin(freqs, freqs.size//nchunk)
        data = rebin(data, nf//nchunk)

    dm = np.zeros(nt)
    dm_err = np.zeros_like(dm)
    dphi_ref = np.zeros_like(dm)
    dphi_ref_err = np.zeros_like(dm)
    amps = np.zeros((nt, nchunk))
    amps_err = np.zeros_like(amps)

    tmplt_f = np.fft.rfft(tmplt, axis=0)
    data_f = np.fft.rfft(data, axis=1)

    freq_ref = np.mean(freqs)

    #TODO SMART GUESS HERE
    for i in range(nt):
        data_f_fitted = data_f[i]
        off_gates = (data[i]<np.median(data[i], axis=0))

        dmguess = (0.0*u.pc/u.cm**3)
        dphi_refguess = template_match(tmplt.mean(-1),data[i].mean(-1))[0][0]

        xguess = [dmguess.decompose(bases=([u.pc,u.cm])).value, dphi_refguess]

        data_var = np.zeros(nchunk)
        for j in range(data_var.size):
                data_var[j] = np.var(data[i,:,j][off_gates[:,j]])

        minchisq = minimize(chi_all, x0=xguess, args=(data_f_fitted, data_var * (nph/2), tmplt_f, ppsr, spin_freqs, freqs, nph, freq_ref), method='Nelder-Mead')
        dm_fit, dphi_ref_fit = minchisq.x
        amps_fit = amps_chan([dm_fit, dphi_ref_fit], data_f_fitted, tmplt_f, ppsr, spin_freqs, freqs, freq_ref)

        dm_fit_err, dphi_ref_fit_err, amps_fit_err = calc_errors([dm_fit, dphi_ref_fit], data_f_fitted, data_var * (nph/2),  tmplt_f, ppsr, spin_freqs, freqs, freq_ref)

        dm[i], dm_err[i] = dm_fit, dm_fit_err
        dphi_ref[i], dphi_ref_err[i] = dphi_ref_fit, dphi_ref_fit_err
        amps[i], amps_err[i] = amps_fit, amps_fit_err
        print(i)

        if shift_data == True:
            data[i] = shift2(disperse(data[i], freqs, -dm_fit*u.pc/u.cm**3, freq_ref, ppsr), -dphi_ref_fit)

    dm, dm_err = dm  * (u.pc/u.cm**3), dm_err * (u.pc/u.cm**3)

    return dm, dm_err, amps, amps_err, data
