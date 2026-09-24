import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
import tempfile
import os


@dataclass
class AudioSegment:
    audio_data: np.ndarray
    sampling_rate: int
    start_time: float
    duration: float
    speaker_id: Optional[str] = None
    is_silent: bool = False


@dataclass
class AudioFeatures:
    mfcc: np.ndarray
    mel_spectrogram: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray


class AudioProcessor:
    def __init__(
        self,
        target_sampling_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 128,
        silence_threshold: float = 0.01
    ):
        self.target_sampling_rate = target_sampling_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.silence_threshold = silence_threshold
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(audio_path, sr=self.target_sampling_rate, mono=True)
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, output_path: str, sr: Optional[int] = None):
        if sr is None:
            sr = self.target_sampling_rate
        sf.write(output_path, audio, sr)
    
    def resample(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        if original_sr != self.target_sampling_rate:
            return librosa.resample(
                y=audio,
                orig_sr=original_sr,
                target_sr=self.target_sampling_rate
            )
        return audio
    
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        audio = self._remove_dc_offset(audio)
        audio = self._normalize(audio)
        audio = self._apply_preemphasis(audio)
        return audio
    
    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        return audio - np.mean(audio)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _apply_preemphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        return signal.lfilter([1, -coeff], [1], audio)
    
    def detect_silence(
        self,
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        return rms < self.silence_threshold
    
    def remove_silence(
        self,
        audio: np.ndarray,
        top_db: int = 30,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        non_silent_intervals = librosa.effects.split(
            y=audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        if len(non_silent_intervals) == 0:
            return np.array([])
        
        audio_segments = []
        for start, end in non_silent_intervals:
            audio_segments.append(audio[start:end])
        
        return np.concatenate(audio_segments) if audio_segments else np.array([])
    
    def enhance_weak_audio(
        self,
        audio: np.ndarray,
        noise_reduction: bool = True,
        gain: float = 2.0,
        adaptive_threshold: float = 0.1
    ) -> np.ndarray:
        if noise_reduction:
            audio = self._spectral_subtraction(audio)
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < adaptive_threshold:
            audio = audio * gain
        
        audio = self._soft_clip(audio)
        
        return audio
    
    def _spectral_subtraction(
        self,
        audio: np.ndarray,
        noise_frames: int = 10,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        if len(audio) < frame_length:
            return audio
        
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        cleaned_magnitude = np.maximum(magnitude - 2 * noise_estimate, 0)
        
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_length)
        
        return cleaned_audio
    
    def _soft_clip(self, audio: np.ndarray, threshold: float = 0.99) -> np.ndarray:
        return np.tanh(audio / threshold) * threshold
    
    def extract_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> AudioFeatures:
        if sr is None:
            sr = self.target_sampling_rate
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc
        )
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels
        )
        
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        rms_energy = librosa.feature.rms(y=audio)
        
        return AudioFeatures(
            mfcc=mfcc,
            mel_spectrogram=mel_spectrogram,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            rms_energy=rms_energy
        )
    
    def detect_voice_activity(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> List[Dict]:
        if sr is None:
            sr = self.target_sampling_rate
        
        features = self.extract_features(audio, sr)
        
        rms = features.rms_energy[0]
        zcr = features.zero_crossing_rate[0]
        centroid = features.spectral_centroid[0]
        
        rms_threshold = np.percentile(rms, 30)
        zcr_threshold = np.percentile(zcr, 70)
        
        voice_segments = []
        current_segment = None
        
        for i in range(len(rms)):
            is_voice = (rms[i] > rms_threshold * 1.5) and (zcr[i] < zcr_threshold)
            
            if is_voice and current_segment is None:
                current_segment = {
                    "start_frame": i,
                    "start_time": i * hop_length / sr,
                    "frames": []
                }
                current_segment["frames"].append(i)
            elif is_voice and current_segment is not None:
                current_segment["frames"].append(i)
            elif not is_voice and current_segment is not None:
                current_segment["end_frame"] = i
                current_segment["end_time"] = i * hop_length / sr
                current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
                
                if current_segment["duration"] > 0.1:
                    voice_segments.append(current_segment)
                
                current_segment = None
        
        if current_segment is not None:
            current_segment["end_frame"] = len(rms) - 1
            current_segment["end_time"] = (len(rms) - 1) * hop_length / sr
            current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
            
            if current_segment["duration"] > 0.1:
                voice_segments.append(current_segment)
        
        return voice_segments
    
    def segment_audio(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        max_segment_duration: float = 30.0
    ) -> List[AudioSegment]:
        if sr is None:
            sr = self.target_sampling_rate
        
        voice_segments = self.detect_voice_activity(audio, sr)
        
        audio_segments = []
        for vs in voice_segments:
            start_sample = int(vs["start_time"] * sr)
            end_sample = int(vs["end_time"] * sr)
            
            segment_audio = audio[start_sample:end_sample]
            segment_duration = end_sample / sr - start_sample / sr
            
            if segment_duration > max_segment_duration:
                sub_segments = self._split_long_segment(
                    segment_audio, sr, max_segment_duration
                )
                audio_segments.extend(sub_segments)
            else:
                audio_segments.append(AudioSegment(
                    audio_data=segment_audio,
                    sampling_rate=sr,
                    start_time=vs["start_time"],
                    duration=segment_duration,
                    is_silent=False
                ))
        
        return audio_segments
    
    def _split_long_segment(
        self,
        audio: np.ndarray,
        sr: int,
        max_duration: float
    ) -> List[AudioSegment]:
        segments = []
        max_samples = int(max_duration * sr)
        start_sample = 0
        
        while start_sample < len(audio):
            end_sample = min(start_sample + max_samples, len(audio))
            segment_audio = audio[start_sample:end_sample]
            
            segments.append(AudioSegment(
                audio_data=segment_audio,
                sampling_rate=sr,
                start_time=start_sample / sr,
                duration=(end_sample - start_sample) / sr,
                is_silent=False
            ))
            
            start_sample = end_sample
        
        return segments
    
    def compute_snr(self, audio: np.ndarray, noise_audio: np.ndarray) -> float:
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    def get_audio_duration(self, audio: np.ndarray, sr: Optional[int] = None) -> float:
        if sr is None:
            sr = self.target_sampling_rate
        return len(audio) / sr
    
    def to_temp_file(self, audio: np.ndarray, sr: Optional[int] = None) -> str:
        if sr is None:
            sr = self.target_sampling_rate
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        sf.write(temp_path, audio, sr)
        
        return temp_path
    
    def cleanup_temp_file(self, file_path: str):
        if os.path.exists(file_path):
            os.unlink(file_path)
