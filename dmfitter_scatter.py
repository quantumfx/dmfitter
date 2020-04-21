from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from scipy.optimize import minimize

k_dm = const.e.gauss**2/(2*np.pi*const.m_e*const.c)

def abs2(x):
    return x.real**2 + x.imag**2


def dphi(dm, ppsr, freqs):
    dphi = (k_dm * (dm * u.pc / u.cm**3) / ppsr * (freqs**(-2))).to(
        u.dimensionless_unscaled)
    return dphi


def shift_factor(dm, ppsr, spin_freqs, freqs):
    """Returns the Fourier shift factor"""
    shift_factor = np.exp(-2j * np.pi * spin_freqs[:, np.newaxis] *
                          dphi(dm, ppsr, freqs))
    return shift_factor


def scatter_factor(tau0, ppsr, spin_freqs, freqs):
    """Returns the Fourier scattering factor"""
    f0 = np.mean(freqs)
    scatter_factor = 1 / (1 + 2j * np.pi * spin_freqs[:, np.newaxis] *
                          (tau0 * u.us / ppsr *
                           (freqs / f0)**(-4)).to(u.dimensionless_unscaled))
    return scatter_factor


def Cxx_chan_data(data_f):
    return np.sum(abs2(data_f)[1:], axis=0)


def Cxx_chan_tmplt(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    #data_f not used, passed for simplicity later on
    dm, tau0 = p
    num = abs2(tmplt_f * scatter_factor(tau0, ppsr, spin_freqs, freqs))
    return np.sum(num[1:], axis=0)


def Cxy_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    num = np.real(data_f *
                  (tmplt_f * shift_factor(dm, ppsr, spin_freqs, freqs) *
                   scatter_factor(tau0, ppsr, spin_freqs, freqs)).conj())
    return np.sum(num[1:], axis=0)


def amps_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    params = [p, data_f, tmplt_f, ppsr, spin_freqs, freqs]
    return Cxy_chan(*params) / Cxx_chan_tmplt(*params)


def chi_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, nph):
    chisq = Cxx_chan_data(data_f) / data_var - Cxy_chan(
        p, data_f, tmplt_f, ppsr, spin_freqs, freqs)**2 / Cxx_chan_tmplt(
            p, data_f, tmplt_f, ppsr, spin_freqs, freqs) / data_var
    return chisq


def chi_all(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs, nph):
    chisq_chan = chi_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs,
                          freqs, nph)
    return np.sum(chisq_chan)


def Cxyd1tau0_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    f0 = np.mean(freqs)
    num = np.real(
        data_f * (tmplt_f * shift_factor(dm, ppsr, spin_freqs, freqs)).conj() *
        scatter_factor(tau0, ppsr, spin_freqs, freqs).conj()**(2) * 2j *
        np.pi * spin_freqs[:, np.newaxis] * (freqs / f0)**(-4) * (u.us/ppsr).to(u.dimensionless_unscaled))
    return np.sum(num[1:], axis=0)


def Cxxd1tau0_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    f0 = np.mean(freqs)
    num = abs2(tmplt_f) * abs2(scatter_factor(
        tau0, ppsr, spin_freqs,
        freqs))**2 * (-2 * tau0 ) * (2 * np.pi * spin_freqs[:, np.newaxis] *
                                    (freqs / f0)**(-4)*(u.us/ppsr).to(u.dimensionless_unscaled))**2
    #     num = abs2(tmplt_f) * abs2(scatter_factor(
    #         tau0, ppsr, spin_freqs, freqs))**2 * scatter_factor(
    #             tau0, ppsr, spin_freqs,
    #             freqs)**(-1) * (-2j) * (2 * np.pi * spin_freqs[:, np.newaxis] *
    #                                     (freqs / f0)**(-4))
    return np.sum(num[1:], axis=0)


def Cxyd2tau0_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    f0 = np.mean(freqs)
    num = np.real(
        data_f * (tmplt_f * shift_factor(dm, ppsr, spin_freqs, freqs)).conj() *
        scatter_factor(tau0, ppsr, spin_freqs, freqs).conj()**(3) * -2 *
        (2 * np.pi * spin_freqs[:, np.newaxis] * (freqs / f0)**(-4)*(u.us/ppsr).to(u.dimensionless_unscaled))**2)
    return np.sum(num[1:], axis=0)


def Cxxd2tau0_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    f0 = np.mean(freqs)
    Bsqr = (2 * np.pi * spin_freqs[:, np.newaxis] * (freqs / f0)**(-4)*(u.us/ppsr).to(u.dimensionless_unscaled))**2
    num = 2 * abs2(tmplt_f) * Bsqr * (-1 + 3 * Bsqr * tau0**2) * abs2(
        scatter_factor(tau0, ppsr, spin_freqs, freqs))**3
    #     num = abs2(tmplt_f) * 2 * (
    #         2 * np.pi * spin_freqs[:, np.newaxis] * (freqs / f0)**(-4))**2 * abs2(
    #             scatter_factor(tau0, ppsr, spin_freqs, freqs)
    #         )**2 - 8 * scatter_factor(tau0, ppsr, spin_freqs, freqs)**(-2) * (
    #             2 * np.pi * spin_freqs[:, np.newaxis] * (freqs / f0)**
    #             (-4))**2 * abs2(scatter_factor(tau0, ppsr, spin_freqs, freqs))**3
    return np.sum(num[1:], axis=0)


def Cxyd1dm_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    num = np.real(data_f *
                  (tmplt_f * shift_factor(dm, ppsr, spin_freqs, freqs) *
                   scatter_factor(tau0, ppsr, spin_freqs, freqs)).conj() * 2j *
                  np.pi * spin_freqs[:, np.newaxis])
    return np.sum(num[1:], axis=0)


def Cxyd2dm_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    num = np.real(data_f *
                  (tmplt_f * shift_factor(dm, ppsr, spin_freqs, freqs) *
                   scatter_factor(tau0, ppsr, spin_freqs, freqs)).conj() *
                  -(2 * np.pi * spin_freqs[:, np.newaxis])**2)
    return np.sum(num[1:], axis=0)


def Cxyddmdtau0_chan(p, data_f, tmplt_f, ppsr, spin_freqs, freqs):
    dm, tau0 = p
    f0 = np.mean(freqs)
    num = np.real(
        data_f * (tmplt_f * shift_factor(dm, ppsr, spin_freqs, freqs)).conj() *
        -(2 * np.pi * spin_freqs[:, np.newaxis])**2 *
        scatter_factor(tau0, ppsr, spin_freqs, freqs).conj()**2 *
        (freqs / f0)**(-4)*(u.us/ppsr).to(u.dimensionless_unscaled))
    return np.sum(num[1:], axis=0)


def d2chisqdtau02_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs):
    params = [p, data_f, tmplt_f, ppsr, spin_freqs, freqs]
    dm, tau0 = p

    num = (-2 * (Cxx_chan_tmplt(*params) * Cxyd1tau0_chan(*params) -
                 Cxy_chan(*params) * Cxxd1tau0_chan(*params))**2 - 2 *
           (Cxy_chan(*params) * Cxx_chan_tmplt(*params)**2 *
            Cxyd2tau0_chan(*params)) +
           Cxy_chan(*params)**2 * Cxx_chan_tmplt(*params) *
           Cxxd2tau0_chan(*params)) / (Cxx_chan_tmplt(*params)**3) / data_var

    return num


def d2chisqdmdtau0_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs):
    params = [p, data_f, tmplt_f, ppsr, spin_freqs, freqs]
    dm, tau0 = p
    num = -((2 * (Cxx_chan_tmplt(*params) * Cxyd1dm_chan(*params) *
                  Cxyd1tau0_chan(*params) + Cxy_chan(*params) *
                  (-Cxxd1tau0_chan(*params) * Cxyd1dm_chan(*params) +
                   Cxx_chan_tmplt(*params) * Cxyddmdtau0_chan(*params)))) /
            (Cxx_chan_tmplt(*params)**2) / data_var)

    return num


def w_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs):
    params = [p, data_f, tmplt_f, ppsr, spin_freqs, freqs]
    return (Cxyd2dm_chan(*params) * Cxy_chan(*params) +
            Cxyd1dm_chan(*params)**2) / Cxx_chan_tmplt(*params) / data_var

def jacobian(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs):
    params = [p, data_f, tmplt_f, ppsr, spin_freqs, freqs]

    wn = w_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs)
    xn = k_dm / ppsr * (freqs**(-2))
    dchisqdm = -2 * np.sum(xn * wn).to(u.cm**3/u.pc).value
    dchisqtau0 = Cxy_chan(*params)*(Cxy_chan(*params)*Cxxd1tau0_chan(*params)-2*Cxx_chan_tmplt(*params)*Cxyd1tau0_chan(*params)) / data_var

    return np.array([dchisqdm,dchisqtau0])

def hessian(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs):
    wn = w_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs)
    xn = k_dm / ppsr * (freqs**(-2))

    d2chisqdtau02 = np.sum(
        d2chisqdtau02_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs,
                           freqs)) * (u.us)**(-2)
    d2chisqddmdtau0 = np.sum(xn * d2chisqdmdtau0_chan(
        p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs)) * (u.us)**(-1)
    d2chisqdm2 = -2 * np.sum(xn**2 * wn)

    return np.array([[d2chisqdm2,d2chisqdmdtau0],[d2chisqdmdtau0,d2chisqdtau02]]).value


def calc_errors(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs):
    wn = w_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs)
    xn = k_dm / ppsr * (freqs**(-2))

    d2chisqdtau02 = np.sum(
        d2chisqdtau02_chan(p, data_f, data_var, tmplt_f, ppsr, spin_freqs,
                           freqs)) * (u.us)**(-2)
    d2chisqddmdtau0 = np.sum(xn * d2chisqdmdtau0_chan(
        p, data_f, data_var, tmplt_f, ppsr, spin_freqs, freqs)) * (u.us)**(-1)
    d2chisqdm2 = -2 * np.sum(xn**2 * wn)
    d2chisqdan2 = 2 * Cxx_chan_tmplt(p, data_f, tmplt_f, ppsr, spin_freqs,
                                     freqs) / data_var

    hess_det = d2chisqdm2 * d2chisqdtau02 - (d2chisqddmdtau0)**2
    an_err = 1 / np.sqrt(d2chisqdan2)
    dm_err = np.sqrt(d2chisqdtau02 / hess_det)
    tau0_err = np.sqrt(d2chisqdm2 / hess_det)

    return dm_err.to(u.pc/u.cm**3).value, an_err, tau0_err.to(u.us).value

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
    2D sub-bin shifting, shifting axis 0
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
        tmplt = None #data[-250:,...].mean(0)
    else:
        tmplt = template
    if nchunk != nf:
        freqs = rebin(freqs, freqs.size//nchunk)
        data = rebin(data, nf//nchunk)

    dm = np.zeros(nt)
    dm_err = np.zeros_like(dm)
    amps = np.zeros((nt, nchunk))
    amps_err = np.zeros_like(amps)
    tau = np.zeros(nt)
    tau_err = np.zeros_like(tau)

    tmplt_f = np.fft.rfft(tmplt, axis=0)
    data_f = np.fft.rfft(data, axis=1)

    freq_ref = np.mean(freqs)

    #TODO SMART GUESS HERE
    for i in range(nt):
        data_f_fitted = data_f[i]
        off_gates = (data[i]<np.median(data[i], axis=0))

        data_var = np.zeros(nchunk)
        for j in range(data_var.size):
                data_var[j] = np.var(data[i,:,j][off_gates[:,j]])
        # sometimes nan sneaks into data_var due to bad channels, which causes the chisq min to not converge. Temporary fix: fill them with mean of the rest of the channels
        # TODO: smarter channel variance
        # data_var = np.nan_to_num(data_var,nan=np.nanmean(data_var))
        data_var[np.isnan(data_var)] = np.nanmean(data_var)

        if i == 0:
            # smart-ish guess for the DM. better will be to use a running average, or something. This fails sometimes
            dmguess = (template_match(tmplt.mean(-1),data[i].mean(-1))[0][0]*ppsr*freqs.mean()**2/k_dm).to(u.pc/u.cm**3)
            tauguess = 1*u.us
        elif i < 5:
            dmguess = dm[i-1]*(u.pc/u.cm**3)
            tauguess = np.abs(tau[i-1])*u.us
        else:
            dmguess = dm[i-5:i].mean() * (u.pc/u.cm**3)
            tauguess = tau[i-5:i-1].mean() * u.us

        xguess = [dmguess.decompose(bases=([u.pc,u.cm])).value, tauguess.decompose(bases=([u.us])).value]

        minchisq = minimize(chi_all, x0=xguess, args=(data_f_fitted, data_var * (nph/2), tmplt_f, ppsr, spin_freqs, freqs, nph), jac=jacobian, hess=hessian, method='BFGS')
        if minchisq.success != True:
            print('Chi square minimization failed to converge at time '+str(i)+' of '+str(nt)+'. !!BEWARE!!')

        fitted_params = minchisq.x
        dm_fit = fitted_params[0]
        amps_fit = amps_chan(fitted_params, data_f_fitted, tmplt_f, ppsr, spin_freqs, freqs)
        tau_fit = fitted_params[1]

        dm_fit_err, amps_fit_err, tau_fit_err = calc_errors(fitted_params, data_f_fitted, data_var * (nph/2),  tmplt_f, ppsr, spin_freqs, freqs)

        dm[i], dm_err[i] = dm_fit, dm_fit_err
        amps[i], amps_err[i] = amps_fit, amps_fit_err
        tau[i], tau_err[i] = tau_fit, tau_fit_err
        print(i)

        if shift_data == True:
            data[i] = disperse(data[i], freqs, -dm_fit*u.pc/u.cm**3, np.inf, ppsr)

    dm, dm_err = dm  * (u.pc/u.cm**3), dm_err * (u.pc/u.cm**3)
    tau, tau_err = tau * (u.us), tau_err * (u.us)

    return dm, dm_err, amps, amps_err, tau, tau_err, data
