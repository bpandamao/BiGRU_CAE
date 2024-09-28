# BGR_CAE
### A novel stacked hybrid autoencoder for imputing LISA data gaps

The Laser Interferometer Space Antenna (LISA) data stream will contain gaps with missing or unusable data due to antenna repointing, orbital corrections, instrument malfunctions, and unknown random processes.  We introduce a new deep learning model to impute data gaps in the LISA data stream.  The stacked hybrid autoencoder combines a denoising convolutional autoencoder (DCAE) with a bi-directional gated recurrent unit (BiGRU).  The DCAE is used to extract relevant features in the corrupted data, while the BiGRU captures the temporal dynamics of the gravitational-wave signals. 


Here, we demonstrate the process of training BGR-CAE for the toy model signal with gaps.

We chose a toy model that describes a chirping waveform.

$h(t;a,f,\dot{f},\epsilon) = a \sin (2\pi t[f + 1/2\dot{f}t])$

| parameter | default_value | training_range|
|-----------|------------|--------------------|
| $\dot{f}$ | $10^{-8}$  | uniform|$10^{-12}$|
| $a$ | $5\cdot 10^{-21}$  |uniform|$10^{-21}$|
| $f$ | $10^{-3}$  | uniform|$10^{-6}$|

## Code structure
**Models**
Contains the method to generate data and gaps, as well as the setting for DCAE and Bi-GRU;

**Training**
Contains two notebooks to train BGR-CAE:



## Get started
1. Install Anaconda if you do not have it.
2. Create a virtual environment using:
```
conda create -n BGR_CAE_trial -c conda-forge numpy scipy matplotlib jupyter torch sklearn random
conda activate BGR_CAE_trial
```
