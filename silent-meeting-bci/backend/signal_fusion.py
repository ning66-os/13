import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class SignalSource(Enum):
    BCI = "bci"
    AUDIO = "audio"
    FUSED = "fused"


@dataclass
class TextSegment:
    text: str
    confidence: float
    timestamp: float
    source: SignalSource
    speaker_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class FusionResult:
    segments: List[TextSegment]
    fused_text: str
    overall_confidence: float


class SignalFusion:
    def __init__(
        self,
        bci_weight: float = 0.4,
        audio_weight: float = 0.6,
        time_window: float = 2.0,
        max_delay: float = 1.0
    ):
        self.bci_weight = bci_weight
        self.audio_weight = audio_weight
        self.time_window = time_window
        self.max_delay = max_delay
        
        self.bci_buffer: List[TextSegment] = []
        self.audio_buffer: List[TextSegment] = []
        self.fused_segments: List[TextSegment] = []
    
    def add_bci_segment(self, segment: TextSegment):
        if segment.source != SignalSource.BCI:
            segment.source = SignalSource.BCI
        self.bci_buffer.append(segment)
    
    def add_audio_segment(self, segment: TextSegment):
        if segment.source != SignalSource.AUDIO:
            segment.source = SignalSource.AUDIO
        self.audio_buffer.append(segment)
    
    def fuse(self) -> FusionResult:
        self._clean_old_segments()
        
        aligned_segments = self._align_timestamps()
        
        fused_segments = []
        
        for bci_seg, audio_seg in aligned_segments:
            fused = self._fuse_pair(bci_seg, audio_seg)
            if fused:
                fused_segments.append(fused)
        
        self.bci_buffer.clear()
        self.audio_buffer.clear()
        
        self.fused_segments.extend(fused_segments)
        
        fused_text = " ".join([seg.text for seg in fused_segments if seg.text])
        
        overall_confidence = 0.0
        if fused_segments:
            confidences = [seg.confidence for seg in fused_segments]
            overall_confidence = float(np.mean(confidences))
        
        return FusionResult(
            segments=fused_segments,
            fused_text=fused_text,
            overall_confidence=overall_confidence
        )
    
    def _clean_old_segments(self):
        timestamps = [seg.timestamp for seg in self.bci_buffer]
        timestamps += [seg.timestamp for seg in self.audio_buffer]
        if not timestamps:
            return
        reference_time = max(timestamps)
        
        self.bci_buffer = [
            seg for seg in self.bci_buffer
            if (reference_time - seg.timestamp) < self.time_window * 10
        ]
        self.audio_buffer = [
            seg for seg in self.audio_buffer
            if (reference_time - seg.timestamp) < self.time_window * 10
        ]
    
    def _align_timestamps(self) -> List[Tuple[Optional[TextSegment], Optional[TextSegment]]]:
        pairs = []
        
        used_audio_indices = set()
        
        for bci_seg in self.bci_buffer:
            best_match = None
            best_diff = float('inf')
            best_idx = -1
            
            for i, audio_seg in enumerate(self.audio_buffer):
                if i in used_audio_indices:
                    continue
                
                time_diff = abs(bci_seg.timestamp - audio_seg.timestamp)
                
                if time_diff < best_diff and time_diff <= self.max_delay:
                    best_diff = time_diff
                    best_match = audio_seg
                    best_idx = i
            
            if best_match and best_idx >= 0:
                used_audio_indices.add(best_idx)
                pairs.append((bci_seg, best_match))
            else:
                pairs.append((bci_seg, None))
        
        for i, audio_seg in enumerate(self.audio_buffer):
            if i not in used_audio_indices:
                pairs.append((None, audio_seg))
        
        return pairs
    
    def _fuse_pair(
        self,
        bci_seg: Optional[TextSegment],
        audio_seg: Optional[TextSegment]
    ) -> Optional[TextSegment]:
        if bci_seg is None and audio_seg is None:
            return None
        
        if bci_seg is None:
            return TextSegment(
                text=audio_seg.text,
                confidence=audio_seg.confidence,
                timestamp=audio_seg.timestamp,
                source=SignalSource.FUSED,
                speaker_id=audio_seg.speaker_id,
                start_time=audio_seg.start_time,
                end_time=audio_seg.end_time
            )
        
        if audio_seg is None:
            return TextSegment(
                text=bci_seg.text,
                confidence=bci_seg.confidence,
                timestamp=bci_seg.timestamp,
                source=SignalSource.FUSED,
                speaker_id=bci_seg.speaker_id,
                start_time=bci_seg.start_time,
                end_time=bci_seg.end_time
            )
        
        fused_text = self._merge_texts(bci_seg.text, audio_seg.text)
        
        fused_confidence = (
            bci_seg.confidence * self.bci_weight +
            audio_seg.confidence * self.audio_weight
        )
        
        speaker_id = audio_seg.speaker_id or bci_seg.speaker_id
        
        bci_start = bci_seg.start_time if bci_seg.start_time is not None else bci_seg.timestamp
        audio_start = audio_seg.start_time if audio_seg.start_time is not None else audio_seg.timestamp
        bci_end = bci_seg.end_time if bci_seg.end_time is not None else bci_seg.timestamp
        audio_end = audio_seg.end_time if audio_seg.end_time is not None else audio_seg.timestamp
        
        return TextSegment(
            text=fused_text,
            confidence=fused_confidence,
            timestamp=min(bci_seg.timestamp, audio_seg.timestamp),
            source=SignalSource.FUSED,
            speaker_id=speaker_id,
            start_time=min(bci_start, audio_start),
            end_time=max(bci_end, audio_end)
        )
    
    def _merge_texts(self, text1: str, text2: str) -> str:
        if not text1 and not text2:
            return ""
        if not text1:
            return text2
        if not text2:
            return text1
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        common = words1.intersection(words2)
        
        if len(common) >= len(words1) * 0.5 or len(common) >= len(words2) * 0.5:
            return text2 if len(text2) > len(text1) else text1
        
        return f"{text1} {text2}"
    
    def get_fused_segments(self) -> List[TextSegment]:
        return self.fused_segments.copy()
    
    def clear_buffers(self):
        self.bci_buffer.clear()
        self.audio_buffer.clear()
        self.fused_segments.clear()


class WeakSignalEnhancer:
    def __init__(
        self,
        noise_floor: float = 0.1,
        enhancement_factor: float = 2.0,
        adaptive_threshold: bool = True
    ):
        self.noise_floor = noise_floor
        self.enhancement_factor = enhancement_factor
        self.adaptive_threshold = adaptive_threshold
        
        self.signal_history: List[float] = []
        
    def enhance(
        self,
        signal: np.ndarray,
        confidence: float
    ) -> Tuple[np.ndarray, float]:
        signal_power = float(np.mean(np.square(signal)))
        
        if signal_power < self.noise_floor:
            if self.adaptive_threshold:
                adaptive_factor = self.enhancement_factor * (self.noise_floor / (signal_power + 1e-10))
                adaptive_factor = min(adaptive_factor, self.enhancement_factor * 3)
            else:
                adaptive_factor = self.enhancement_factor
            
            enhanced_signal = signal * adaptive_factor
            enhanced_confidence = max(0.0, confidence * (1 - (adaptive_factor - 1) * 0.1))
        else:
            enhanced_signal = signal
            enhanced_confidence = confidence
        
        self.signal_history.append(signal_power)
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
        
        return enhanced_signal, enhanced_confidence
    
    def get_average_signal_power(self) -> float:
        if not self.signal_history:
            return 0.0
        return float(np.mean(self.signal_history))


class MultiModalFusion:
    def __init__(self):
        self.modalities: Dict[str, List[TextSegment]] = defaultdict(list)
        self.fusion_weights: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def add_modality(self, name: str, segments: List[TextSegment], weight: float = 1.0):
        self.modalities[name].extend(segments)
        self.fusion_weights[name] = weight
    
    def fuse_all(self) -> FusionResult:
        all_segments = []
        
        total_weight = sum(self.fusion_weights[modality] for modality in self.modalities)
        if total_weight <= 0:
            total_weight = 1.0
        
        for modality, segments in self.modalities.items():
            weight = self.fusion_weights[modality] / total_weight
            for seg in segments:
                weighted_seg = TextSegment(
                    text=seg.text,
                    confidence=seg.confidence * weight,
                    timestamp=seg.timestamp,
                    source=seg.source,
                    speaker_id=seg.speaker_id,
                    start_time=seg.start_time,
                    end_time=seg.end_time
                )
                all_segments.append(weighted_seg)
        
        all_segments.sort(key=lambda x: x.timestamp)
        
        merged = self._merge_overlapping(all_segments)
        
        fused_text = " ".join([seg.text for seg in merged if seg.text])
        
        overall_confidence = 0.0
        if merged:
            confidences = [seg.confidence for seg in merged]
            overall_confidence = float(np.mean(confidences))
        
        return FusionResult(
            segments=merged,
            fused_text=fused_text,
            overall_confidence=overall_confidence
        )
    
    def _merge_overlapping(self, segments: List[TextSegment]) -> List[TextSegment]:
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            if seg.timestamp - last.timestamp < 1.0 and last.speaker_id == seg.speaker_id:
                merged_text = f"{last.text} {seg.text}".strip()
                merged_conf = (last.confidence + seg.confidence) / 2
                
                last_end = last.end_time if last.end_time is not None else last.timestamp
                seg_end = seg.end_time if seg.end_time is not None else seg.timestamp
                
                merged[-1] = TextSegment(
                    text=merged_text,
                    confidence=merged_conf,
                    timestamp=last.timestamp,
                    source=SignalSource.FUSED,
                    speaker_id=last.speaker_id,
                    start_time=last.start_time,
                    end_time=max(last_end, seg_end)
                )
            else:
                merged.append(seg)
        
        return merged
    
    def clear(self):
        self.modalities.clear()
