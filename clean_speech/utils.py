from scipy import signal
import numpy as np
import soundfile as sf

MAX_CHUNK_LENGTH = 16384
FS = 16000
MIN_AMP = 10000.
AMP_FAC = 10000.
X_MEAN = 7.6765018
X_STD = 17.440527

def db(x, min_amp, amp_fac):
    return 20. * np.log10(np.maximum(x, np.max(x) / min_amp) * amp_fac)
    
def reduce_standarize(x, x_mean, x_std):
    return (x - x_mean) / x_std

def feature_extractor(data, nperseg = 512, noverlap = 256):
    
    real_length = len(data)

    residual = real_length % MAX_CHUNK_LENGTH 

    pad = np.zeros((MAX_CHUNK_LENGTH -  residual,))

    data = np.concatenate([data,pad])

    data = data.reshape(-1,MAX_CHUNK_LENGTH)

    X = None
    complex_array = None

    for ch in data:
        
        _, _, complex_ = signal.stft(ch, fs=FS, nperseg=nperseg, noverlap=noverlap)
        
        complex_ = complex_.T[np.newaxis,...]
        
        mag = np.abs(complex_)

        x = reduce_standarize(db(mag,MIN_AMP,AMP_FAC),X_MEAN,X_STD)

        if X is None:

            X = np.zeros((0,x.shape[1],x.shape[2]))
        
        if complex_array is None:

            complex_array = np.zeros((0,complex_.shape[1],complex_.shape[2]))

        X = np.append(X,x,axis=0)

        complex_array = np.append(complex_array,complex_,axis=0)

    return X[...,np.newaxis], complex_array[...,np.newaxis]


def convert_to_audiowave(results,complex_, nperseg = 512, noverlap = 256):
        
        recovered_a = []

        for predicted_, complex_mix in zip(results,complex_):
            
            _, source_recovered_a = signal.istft((predicted_[...,0] * complex_mix[...,0]).T, fs=FS ,nperseg= nperseg, noverlap = noverlap)
            
            recovered_a.append(source_recovered_a)
        
        return np.array(recovered_a).reshape(-1)
    