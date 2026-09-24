import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


@dataclass
class SpeakerSegment:
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float
    features: Optional[np.ndarray] = None


@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment]
    speaker_count: int
    unique_speakers: List[str]


class EEGSpeakerDiarizer:
    def __init__(
        self,
        num_speakers: Optional[int] = None,
        feature_dim: int = 64,
        clustering_method: str = "kmeans",
        min_segment_duration: float = 1.0
    ):
        self.num_speakers = num_speakers
        self.feature_dim = feature_dim
        self.clustering_method = clustering_method
        self.min_segment_duration = min_segment_duration
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(feature_dim, 20))
        self._speaker_profiles: Dict[str, np.ndarray] = {}
        self._current_segment_buffer: List[Dict] = []
    
    def extract_speaker_features(
        self,
        eeg_data: np.ndarray,
        sampling_rate: int
    ) -> np.ndarray:
        features = []
        
        n_channels = eeg_data.shape[0]
        
        for ch in range(n_channels):
            channel_data = eeg_data[ch]
            
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
            features.append(np.median(channel_data))
            features.append(np.min(channel_data))
            features.append(np.max(channel_data))
            
            features.append(np.percentile(channel_data, 25))
            features.append(np.percentile(channel_data, 75))
            
            features.append(np.mean(np.abs(np.diff(channel_data))))
            
            zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
            features.append(zero_crossings / len(channel_data))
        
        freq_features = self._extract_frequency_features(eeg_data, sampling_rate)
        features.extend(freq_features)
        
        connectivity_features = self._extract_connectivity_features(eeg_data)
        features.extend(connectivity_features)
        
        return np.array(features)
    
    def _extract_frequency_features(
        self,
        eeg_data: np.ndarray,
        sampling_rate: int
    ) -> List[float]:
        from scipy import signal
        
        features = []
        freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 40)
        }
        
        n_channels = eeg_data.shape[0]
        
        for ch in range(n_channels):
            freqs, psd = signal.welch(eeg_data[ch], fs=sampling_rate, nperseg=256)
            
            band_powers = []
            for low, high in freq_bands.values():
                mask = (freqs >= low) & (freqs <= high)
                power = np.mean(psd[mask]) if np.any(mask) else 0
                band_powers.append(power)
            
            total_power = np.sum(band_powers) + 1e-10
            relative_powers = [p / total_power for p in band_powers]
            
            features.extend(band_powers)
            features.extend(relative_powers)
            
            if total_power > 1e-10:
                spectral_entropy = -np.sum(
                    (np.array(band_powers) / total_power) *
                    np.log2(np.array(band_powers) / total_power + 1e-10)
                )
                features.append(spectral_entropy)
            else:
                features.append(0)
        
        return features
    
    def _extract_connectivity_features(self, eeg_data: np.ndarray) -> List[float]:
        features = []
        n_channels = eeg_data.shape[0]
        
        if n_channels < 2:
            return [0] * 10
        
        corr_matrix = np.corrcoef(eeg_data)
        
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        features.append(np.mean(upper_tri))
        features.append(np.std(upper_tri))
        features.append(np.max(upper_tri))
        features.append(np.min(upper_tri))
        
        symmetric = (corr_matrix + corr_matrix.T) / 2
        eigenvalues = np.linalg.eigvalsh(symmetric)
        features.extend(eigenvalues[:5].tolist())
        
        return features
    
    def diarize(
        self,
        eeg_segments: List[Dict],
        num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        if not eeg_segments:
            return DiarizationResult(segments=[], speaker_count=0, unique_speakers=[])
        
        if num_speakers is not None:
            self.num_speakers = num_speakers
        
        all_features = []
        valid_segments = []
        
        for seg in eeg_segments:
            eeg_data = seg.get("eeg_data")
            sampling_rate = seg.get("sampling_rate", 250)
            
            if eeg_data is None or eeg_data.size == 0:
                continue
            
            try:
                features = self.extract_speaker_features(eeg_data, sampling_rate)
                all_features.append(features)
                valid_segments.append(seg)
            except Exception as e:
                continue
        
        if not all_features:
            return DiarizationResult(segments=[], speaker_count=0, unique_speakers=[])
        
        features_matrix = np.array(all_features)
        
        features_scaled = self.scaler.fit_transform(features_matrix)
        
        try:
            features_reduced = self.pca.fit_transform(features_scaled)
        except Exception as e:
            features_reduced = features_scaled
        
        labels = self._cluster(features_reduced)
        
        speaker_segments = []
        for i, (seg, label) in enumerate(zip(valid_segments, labels)):
            speaker_id = f"speaker_{label}"
            segment = SpeakerSegment(
                start_time=seg.get("start_time", 0),
                end_time=seg.get("end_time", 0),
                speaker_id=speaker_id,
                confidence=seg.get("confidence", 0.5),
                features=features_reduced[i]
            )
            speaker_segments.append(segment)
        
        speaker_segments.sort(key=lambda x: x.start_time)
        
        merged_segments = self._merge_consecutive_speakers(speaker_segments)
        
        unique_speakers = list(set([seg.speaker_id for seg in merged_segments]))
        
        return DiarizationResult(
            segments=merged_segments,
            speaker_count=len(unique_speakers),
            unique_speakers=unique_speakers
        )
    
    def _cluster(self, features: np.ndarray) -> np.ndarray:
        n_samples = len(features)
        
        if self.num_speakers is None:
            self.num_speakers = min(5, max(2, int(np.sqrt(n_samples))))
        
        if self.clustering_method == "kmeans":
            n_clusters = min(self.num_speakers, n_samples)
            if n_clusters < 2:
                n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
        elif self.clustering_method == "dbscan":
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            labels = dbscan.fit_predict(features)
            
            if -1 in labels:
                noise_mask = labels == -1
                n_noise = np.sum(noise_mask)
                if n_noise > 0 and n_noise < len(labels) // 2:
                    from sklearn.neighbors import KNeighborsClassifier
                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(features[~noise_mask], labels[~noise_mask])
                    labels[noise_mask] = knn.predict(features[noise_mask])
        else:
            labels = np.zeros(n_samples, dtype=int)
        
        return labels
    
    def _merge_consecutive_speakers(
        self,
        segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            if (seg.speaker_id == last.speaker_id and
                (seg.start_time - last.end_time) < 0.5):
                merged[-1] = SpeakerSegment(
                    start_time=last.start_time,
                    end_time=max(last.end_time, seg.end_time),
                    speaker_id=last.speaker_id,
                    confidence=(last.confidence + seg.confidence) / 2,
                    features=last.features
                )
            else:
                merged.append(seg)
        
        filtered = [
            seg for seg in merged
            if (seg.end_time - seg.start_time) >= self.min_segment_duration
        ]
        
        return filtered if filtered else merged
    
    def register_speaker(self, speaker_id: str, eeg_data: np.ndarray, sampling_rate: int):
        features = self.extract_speaker_features(eeg_data, sampling_rate)
        self._speaker_profiles[speaker_id] = features
    
    def identify_speaker(self, eeg_data: np.ndarray, sampling_rate: int) -> Optional[str]:
        if not self._speaker_profiles:
            return None
        
        features = self.extract_speaker_features(eeg_data, sampling_rate)
        
        best_speaker = None
        best_similarity = -1
        
        for speaker_id, profile in self._speaker_profiles.items():
            similarity = self._cosine_similarity(features, profile)
            if similarity > best_similarity and similarity > 0.5:
                best_similarity = similarity
                best_speaker = speaker_id
        
        return best_speaker
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def real_time_diarize(
        self,
        eeg_data: np.ndarray,
        sampling_rate: int,
        timestamp: float,
        window_size: float = 2.0
    ) -> Optional[SpeakerSegment]:
        features = self.extract_speaker_features(eeg_data, sampling_rate)
        
        self._current_segment_buffer.append({
            "timestamp": timestamp,
            "features": features,
            "duration": window_size
        })
        
        if len(self._current_segment_buffer) < 5:
            return None
        
        recent_features = np.array([s["features"] for s in self._current_segment_buffer])
        
        try:
            features_scaled = self.scaler.fit_transform(recent_features)
            labels = self._cluster(features_scaled)
            
            current_label = labels[-1]
            speaker_id = f"speaker_{current_label}"
            
            return SpeakerSegment(
                start_time=timestamp - window_size,
                end_time=timestamp,
                speaker_id=speaker_id,
                confidence=0.7
            )
        except Exception as e:
            return None
    
    def reset(self):
        self._current_segment_buffer.clear()
        self._speaker_profiles.clear()


class AudioSpeakerDiarizer:
    def __init__(
        self,
        auth_token: Optional[str] = None,
        use_pyannote: bool = False
    ):
        self.auth_token = auth_token
        self.use_pyannote = use_pyannote
        self._pipeline = None
        
        if use_pyannote and auth_token:
            self._init_pyannote()
    
    def _init_pyannote(self):
        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )
        except Exception as e:
            self.use_pyannote = False
            self._pipeline = None
    
    def diarize_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        if self.use_pyannote and self._pipeline:
            return self._diarize_with_pyannote(audio_path, num_speakers)
        else:
            return self._diarize_simple(audio_path)
    
    def _diarize_with_pyannote(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        try:
            if num_speakers:
                diarization = self._pipeline(audio_path, num_speakers=num_speakers)
            else:
                diarization = self._pipeline(audio_path)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    start_time=turn.start,
                    end_time=turn.end,
                    speaker_id=f"speaker_{speaker}",
                    confidence=0.9
                ))
            
            unique_speakers = list(set([seg.speaker_id for seg in segments]))
            
            return DiarizationResult(
                segments=segments,
                speaker_count=len(unique_speakers),
                unique_speakers=unique_speakers
            )
        except Exception as e:
            return self._diarize_simple(audio_path)
    
    def _diarize_simple(self, audio_path: str) -> DiarizationResult:
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        duration = len(audio) / sr
        segment_duration = 2.0
        n_segments = int(duration / segment_duration)
        
        segments = []
        for i in range(n_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, duration)
            
            segments.append(SpeakerSegment(
                start_time=start,
                end_time=end,
                speaker_id="speaker_0",
                confidence=0.5
            ))
        
        return DiarizationResult(
            segments=segments,
            speaker_count=1,
            unique_speakers=["speaker_0"]
        )
