### generate 1000 signals with ln(m1) uniform in ln(1e6+-1e5)
import numpy as np
from LISA_utils import FFT, freq_PSD, inner_prod

# from lisatools.sensitivity import get_sensitivity

# set random seed
np.random.seed(1234)

### waveform loading

def waveform(params,t):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function.
    We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    a = params[0]
    f = params[1]
    fdot = params[2]

    return (a *(np.sin((2*np.pi)*(f*t + 0.5*fdot * t**2))))


# Set parameters

ss=1000
a = 5e-21+(1e-21)*np.random.uniform(-1,1,ss)
f = 1e-3+(1e-6)*np.random.uniform(-1,1,ss)
fdot = 1e-8+(1e-12)*np.random.uniform(-1,1,ss)
sig_toy=np.zeros((ss, 51840))
snr=np.zeros(ss)

tmax =  3*24*60*60                 # Final time
delta_t = 5       # Sampling interval

t = np.arange(0,tmax,delta_t)     # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include t = zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series
    
freq,PSD = freq_PSD(t,delta_t)  # Extract frequency bins and PSD.
    
for i in range(ss):
    true_params = [a[i],f[i],fdot[i]]
    signal_t= waveform(true_params,t)
    signal_f= FFT(signal_t)
    SNR2 = inner_prod(signal_f,signal_f,PSD,delta_t,N_t)
    if (i+1)%100==0:
        print("{i} sets have saved!".format(i=i+1))
    sig_toy[i]=signal_t
    snr[i]=np.sqrt(SNR2)
        
print(signal_f.shape)


variance_noise_f = N_t * PSD / (4 * delta_t)            # Calculate variance of noise, real and imaginary.

print(snr)

np.save("./sig_toy_variance.npy",variance_noise_f)
np.save("./sig_toy_signal.npy",sig_toy)
np.save("./snr.npy",snr)
np.save("./a.npy",a)
np.save("./f.npy",f)
np.save("./fdotv.npy",fdot)
