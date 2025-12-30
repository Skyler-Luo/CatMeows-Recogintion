"""
MFCC Feature Extraction Module
"""
import numpy as np
import librosa


def extract_mfcc(y, sr=22050, n_mfcc=13, n_mels=40, n_fft=1024,
                 win_length=None, hop_length=None, window='hamming',
                 fmin=0, fmax=4000,
                 include_delta=True, include_delta2=True, include_delta3=False):
    """
    Extract MFCC features
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel filterbanks
        n_fft: FFT size
        win_length: Window length, None for 30ms
        hop_length: Hop length, None for 20ms
        window: Window function type
        fmin: Minimum frequency
        fmax: Maximum frequency
        include_delta: Include first-order delta
        include_delta2: Include second-order delta
        include_delta3: Include third-order delta
        
    Returns:
        MFCC feature matrix (n_features, n_frames)
    """
    if win_length is None:
        win_length = int(0.030 * sr)  # 30ms
    if hop_length is None:
        hop_length = int(0.020 * sr)  # 20ms
    
    if n_fft < win_length:
        n_fft = 2 ** int(np.ceil(np.log2(win_length)))
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=n_mfcc)
    
    features = [mfcc]
    
    if include_delta:
        mfcc_delta = librosa.feature.delta(mfcc)
        features.append(mfcc_delta)
    
    if include_delta2:
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(mfcc_delta2)
    
    if include_delta3:
        mfcc_delta3 = librosa.feature.delta(mfcc, order=3)
        features.append(mfcc_delta3)
    
    return np.vstack(features)


def _apply_pooling(mfcc, pooling):
    """Apply pooling strategy"""
    pooling_ops = {
        'mean': lambda x: np.mean(x, axis=1),
        'std': lambda x: np.std(x, axis=1),
        'both': lambda x: np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)]),
        'stats': lambda x: np.concatenate([
            np.mean(x, axis=1), np.std(x, axis=1),
            np.min(x, axis=1), np.max(x, axis=1)
        ])
    }
    if pooling not in pooling_ops:
        raise ValueError(f"Unknown pooling method: {pooling}")
    return pooling_ops[pooling](mfcc)


def extract_mfcc_features(audio_list, sr=22050, n_mfcc=13, n_mels=40,
                          pooling='stats', include_delta=True,
                          include_delta2=True, include_delta3=False):
    """
    Batch extract MFCC features with pooling
    
    Args:
        audio_list: List of audio signals
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel filterbanks
        pooling: Pooling method ('mean', 'std', 'both', 'stats')
        include_delta: Include first-order delta
        include_delta2: Include second-order delta
        include_delta3: Include third-order delta
        
    Returns:
        Feature matrix (n_samples, n_features)
    """
    return np.array([
        _apply_pooling(
            extract_mfcc(y, sr, n_mfcc, n_mels,
                         include_delta=include_delta,
                         include_delta2=include_delta2,
                         include_delta3=include_delta3),
            pooling
        )
        for y in audio_list
    ])
