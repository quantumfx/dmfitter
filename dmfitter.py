from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.stats import median_absolute_deviation as mad
from scipy.optimize import minimize

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

def tag_outliers(data, window_size, axis, sigma = 10):
    assert window_size%2 == 0

    data_mad = np.zeros_like(data)
    data_med = np.zeros_like(data)
    #data_tag = np.zeros_like(data).astype(bool)

    pad_width = np.zeros( (len(data.shape), 2) ).astype(int)
    pad_width[axis] = (window_size//2, window_size//2)

    data_pad = np.pad(data, pad_width, mode = 'constant', constant_values = np.nan)

    # compute rolling MAD and median
    for i in range(data_mad.shape[0]):
        data_mad[i] = mad(data[i:i+window_size], axis = axis, ignore_nan = True)
        data_med[i] = np.nanmedian(data[i:i+window_size], axis = axis)

    # tag
    data_tag = np.abs(data - data_med)/data_mad > sigma * 0.67449
#     print('Rejecting MAD >', sigma * 0.67449)
#     print('Total rejected in each IF:', data_tag.sum(0))
    return data_tag

def fit_dm(data, freqs, nchunk, template = None, ppsr = None, shift_data = True, tag_delay_outliers = True, outlier_window = 10):

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

    tag_delay_outliers :
        whether to discard outliers in delay before computing the DM.

    outlier_window :
        size of rolling window in time_bins over which delay outliers are computed.

    Output
    ------

    dm :
        DM in pc/cm^3

    dm_err :
        Error in DM in pc/cm^3

    data :
        shifted data if data_shift is True, otherwise original data.
    """
    assert freqs.size == data.shape[-1]
    assert data.shape[-1]%nchunk == 0

    if ppsr == None:
        print('Please set pulsar period!')
        return

    k_DM = const.e.gauss**2/(2*np.pi*const.m_e*const.c)
    if np.shape(template) == ():
        tmplt = data.mean(0).mean(-1)
    else:
        tmplt = template
    off_gates = (tmplt<np.median(tmplt))
    if nchunk != data.shape[-1]:
        freqs = rebin(freqs, freqs.size//nchunk)
        data = rebin(data, data.shape[-1]//nchunk)

    delay = np.zeros((data.shape[0],nchunk))
    error = np.zeros_like(delay)
    amps = np.zeros_like(delay)
    ampserr = np.zeros_like(delay)

    for i in range(data.shape[0]):
        for j in range(nchunk):
            dt, dterr, a, aerr, b = template_match(tmplt,data[i,:,j],off_gates=off_gates)
            if shift_data == True:
                data[i,:,j] = shift(data[i,:,j],-dt)
            delay[i,j] = dt
            error[i,j] = dterr
            amps[i,j] = a
            ampserr[i,j] = aerr

    if tag_delay_outliers == True:
        delay_outliers = tag_outliers(delay, window_size = outlier_window, axis = 0, sigma = 10)
        delay[delay_outliers] = np.nan
        error[delay_outliers] = np.nan

    delay, error = delay*ppsr, error*ppsr

    dm  = (delay*freqs[np.newaxis,:]**2/k_DM).to(u.pc/u.cm**3)
    dm_err = (error*freqs[np.newaxis,:]**2/k_DM).to(u.pc/u.cm**3)

    weight_tot = np.nansum(1/dm_err**2, axis = -1)
    dm = np.nansum(dm/dm_err**2, axis = -1) / weight_tot
    dm_err = 1/np.sqrt(weight_tot)

    return dm, dm_err, data
