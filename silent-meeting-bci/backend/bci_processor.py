import numpy as np
from scipy import signal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
from collections import deque
from enum import Enum


class BCIChannelType(Enum):
    EEG = "eeg"
    EMG = "emg"
    EOG = "eog"


@dataclass
class BCISignal:
    timestamp: float
    channel_data: np.ndarray
    channel_names: List[str]
    sampling_rate: int


@dataclass
class DecodedWord:
    word: str
    confidence: float
    timestamp: float
    speaker_id: Optional[str] = None


class BCIProcessor:
    def __init__(
        self,
        sampling_rate: int = 250,
        num_channels: int = 8,
        buffer_size: int = 1000
    ):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.signal_buffer: deque = deque(maxlen=buffer_size)
        self.channel_names = [f"EEG_{i+1}" for i in range(num_channels)]
        
        self._init_filters()
        self._init_word_decoder()
    
    def _init_filters(self):
        nyquist = self.sampling_rate / 2
        
        self.bandpass_low = 0.5
        self.bandpass_high = 40.0
        self.notch_freq = 60.0
        
        self.b_bandpass, self.a_bandpass = signal.butter(
            4,
            [self.bandpass_low / nyquist, self.bandpass_high / nyquist],
            btype='band'
        )
        
        self.b_notch, self.a_notch = signal.iirnotch(
            self.notch_freq / nyquist,
            30.0
        )
    
    def _init_word_decoder(self):
        self.word_vocabulary = [
            "yes", "no", "hello", "goodbye", "help",
            "water", "food", "rest", "pain", "happy",
            "sad", "tired", "need", "want", "go",
            "stop", "more", "less", "now", "later",
            "会议", "同意", "反对", "建议", "报告",
            "问题", "解决方案", "讨论", "决定", "行动"
        ]
        
        self.feature_extractor = EEGFeatureExtractor(
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names
        )
    
    def process_raw_signal(self, raw_data: np.ndarray) -> BCISignal:
        if raw_data.shape[0] != self.num_channels:
            raw_data = raw_data.T
        
        filtered_data = self._apply_filters(raw_data)
        
        return BCISignal(
            timestamp=time.time(),
            channel_data=filtered_data,
            channel_names=self.channel_names,
            sampling_rate=self.sampling_rate
        )
    
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        filtered = signal.filtfilt(self.b_bandpass, self.a_bandpass, data, axis=1)
        filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered, axis=1)
        return filtered
    
    def extract_features(self, signal: BCISignal) -> Dict:
        return self.feature_extractor.extract_features(signal.channel_data)
    
    def decode_silent_speech(self, signal: BCISignal) -> Optional[DecodedWord]:
        features = self.extract_features(signal)
        
        word, confidence = self._predict_word(features)
        
        if confidence > 0.3:
            return DecodedWord(
                word=word,
                confidence=confidence,
                timestamp=signal.timestamp
            )
        return None
    
    def _predict_word(self, features: Dict) -> Tuple[str, float]:
        feature_vector = features.get("band_power", np.zeros(5))
        complexity = features.get("complexity", 0)
        asymmetry = features.get("asymmetry", np.zeros(5))
        
        pattern_score = np.abs(feature_vector).mean()
        confidence = min(0.9, 0.3 + pattern_score * 0.5 + complexity * 0.2)
        
        if confidence < 0.3:
            return "", 0.0
        
        vocab_index = int((np.abs(feature_vector).sum() * 100) % len(self.word_vocabulary))
        word = self.word_vocabulary[vocab_index]
        
        return word, confidence
    
    def get_signal_quality(self, signal: BCISignal) -> float:
        data = signal.channel_data
        noise_level = np.mean(np.std(data, axis=1))
        quality = 1.0 - min(1.0, noise_level / 100.0)
        return max(0.0, quality)
    
    def detect_artifacts(self, signal: BCISignal) -> List[str]:
        artifacts = []
        data = signal.channel_data
        
        peak_to_peak = np.max(data, axis=1) - np.min(data, axis=1)
        if np.any(peak_to_peak > 200):
            artifacts.append("eye_blink")
        
        if np.any(np.abs(data) > 500):
            artifacts.append("amplifier_saturation")
        
        high_freq_noise = np.mean(np.abs(np.diff(data, axis=1)))
        if high_freq_noise > 50:
            artifacts.append("high_frequency_noise")
        
        return artifacts


class EEGFeatureExtractor:
    def __init__(self, sampling_rate: int, channel_names: List[str]):
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names
        self.frequency_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 40)
        }
    
    def extract_features(self, eeg_data: np.ndarray) -> Dict:
        features = {}
        
        features["band_power"] = self._compute_band_power(eeg_data)
        features["relative_power"] = self._compute_relative_power(eeg_data)
        features["complexity"] = self._compute_complexity(eeg_data)
        features["asymmetry"] = self._compute_asymmetry(eeg_data)
        features["connectivity"] = self._compute_connectivity(eeg_data)
        
        return features
    
    def _compute_band_power(self, data: np.ndarray) -> np.ndarray:
        freqs, psd = signal.welch(
            data,
            fs=self.sampling_rate,
            nperseg=min(256, data.shape[1])
        )
        
        band_powers = []
        for (low, high) in self.frequency_bands.values():
            mask = (freqs >= low) & (freqs <= high)
            power = np.mean(psd[:, mask], axis=1)
            band_powers.append(np.mean(power))
        
        return np.array(band_powers)
    
    def _compute_relative_power(self, data: np.ndarray) -> np.ndarray:
        band_power = self._compute_band_power(data)
        total_power = np.sum(band_power) + 1e-10
        return band_power / total_power
    
    def _compute_complexity(self, data: np.ndarray) -> float:
        std = np.std(data, axis=1)
        mean_std = np.mean(std)
        return min(1.0, mean_std / 50.0)
    
    def _compute_asymmetry(self, data: np.ndarray) -> np.ndarray:
        n_channels = data.shape[0]
        if n_channels < 2:
            return np.zeros(5)
        
        half = n_channels // 2
        left = data[:half]
        right = data[half:2*half]
        
        asymmetry = []
        for (low, high) in self.frequency_bands.values():
            freqs, psd_left = signal.welch(left, fs=self.sampling_rate)
            _, psd_right = signal.welch(right, fs=self.sampling_rate)
            
            mask = (freqs >= low) & (freqs <= high)
            power_left = np.mean(psd_left[:, mask])
            power_right = np.mean(psd_right[:, mask])
            
            asym = (power_left - power_right) / (power_left + power_right + 1e-10)
            asymmetry.append(asym)
        
        return np.array(asymmetry)
    
    def _compute_connectivity(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] < 2:
            return np.array([])
        
        corr_matrix = np.corrcoef(data)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        return upper_tri


class SilentSpeechDecoder:
    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate
        self.bci_processor = BCIProcessor(sampling_rate=sampling_rate)
        self.word_buffer: List[DecodedWord] = []
        self.max_buffer_size = 100
    
    def process_eeg_packet(self, raw_data: np.ndarray) -> Optional[DecodedWord]:
        signal = self.bci_processor.process_raw_signal(raw_data)
        
        quality = self.bci_processor.get_signal_quality(signal)
        if quality < 0.3:
            return None
        
        artifacts = self.bci_processor.detect_artifacts(signal)
        if artifacts:
            return None
        
        decoded_word = self.bci_processor.decode_silent_speech(signal)
        
        if decoded_word:
            self.word_buffer.append(decoded_word)
            if len(self.word_buffer) > self.max_buffer_size:
                self.word_buffer.pop(0)
        
        return decoded_word
    
    def get_recent_words(self, n: int = 10) -> List[DecodedWord]:
        return self.word_buffer[-n:]
    
    def clear_buffer(self):
        self.word_buffer.clear()
