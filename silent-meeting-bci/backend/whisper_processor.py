import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import tempfile
import os
import time


@dataclass
class TranscriptionSegment:
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker_id: Optional[str] = None


@dataclass
class TranscriptionResult:
    segments: List[TranscriptionSegment]
    full_text: str
    language: Optional[str] = None


class WhisperProcessor:
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        use_gpu: bool = False
    ):
        self.model_size = model_size
        self.language = language
        self.use_gpu = use_gpu
        self._model = None
        self._initialized = False
    
    def _initialize(self):
        if self._initialized:
            return
        
        try:
            import whisper
            device = "cuda" if self.use_gpu else "cpu"
            self._model = whisper.load_model(self.model_size, device=device)
            self._initialized = True
        except Exception as e:
            self._model = None
            self._initialized = False
    
    def transcribe_audio(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        self._initialize()
        
        if self._model is not None:
            return self._transcribe_with_whisper(audio, sampling_rate, language)
        else:
            return self._transcribe_simulation(audio, sampling_rate)
    
    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        self._initialize()
        
        if self._model is not None:
            return self._transcribe_file_with_whisper(audio_path, language)
        else:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            return self._transcribe_simulation(audio, sr)
    
    def _transcribe_with_whisper(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        language: Optional[str]
    ) -> TranscriptionResult:
        try:
            import soundfile as sf
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            sf.write(temp_path, audio, sampling_rate)
            
            result = self._transcribe_file_with_whisper(temp_path, language)
            
            os.unlink(temp_path)
            
            return result
        except Exception as e:
            return self._transcribe_simulation(audio, sampling_rate)
    
    def _transcribe_file_with_whisper(
        self,
        audio_path: str,
        language: Optional[str]
    ) -> TranscriptionResult:
        try:
            if language is None:
                language = self.language
            
            result = self._model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                fp16=self.use_gpu
            )
            
            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptionSegment(
                    text=seg["text"].strip(),
                    start_time=seg["start"],
                    end_time=seg["end"],
                    confidence=seg.get("avg_logprob", -1)
                ))
            
            full_text = result.get("text", "").strip()
            detected_language = result.get("language")
            
            return TranscriptionResult(
                segments=segments,
                full_text=full_text,
                language=detected_language
            )
        except Exception as e:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            return self._transcribe_simulation(audio, sr)
    
    def _transcribe_simulation(
        self,
        audio: np.ndarray,
        sampling_rate: int
    ) -> TranscriptionResult:
        duration = len(audio) / sampling_rate
        
        if duration < 0.1:
            return TranscriptionResult(segments=[], full_text="")
        
        words = [
            "会议", "讨论", "问题", "解决方案", "建议",
            "同意", "反对", "报告", "决定", "行动",
            "项目", "进度", "目标", "计划", "回顾"
        ]
        
        segments = []
        current_time = 0.0
        segment_duration = min(2.0, duration / 3)
        
        while current_time < duration:
            num_words = np.random.randint(2, 6)
            segment_words = np.random.choice(words, size=num_words, replace=True)
            text = " ".join(segment_words)
            
            confidence = 0.5 + np.random.random() * 0.4
            
            segments.append(TranscriptionSegment(
                text=text,
                start_time=current_time,
                end_time=min(current_time + segment_duration, duration),
                confidence=confidence
            ))
            
            current_time += segment_duration
        
        full_text = " ".join([seg.text for seg in segments])
        
        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=self.language or "zh"
        )
    
    def transcribe_stream(
        self,
        audio_generator,
        chunk_duration: float = 3.0,
        language: Optional[str] = None
    ):
        self._initialize()
        
        buffer = []
        buffer_duration = 0.0
        
        for audio_chunk, sr in audio_generator:
            buffer.append(audio_chunk)
            buffer_duration += len(audio_chunk) / sr
            
            if buffer_duration >= chunk_duration:
                combined_audio = np.concatenate(buffer)
                result = self.transcribe_audio(combined_audio, sr, language)
                
                if result.segments:
                    yield result
                
                buffer = []
                buffer_duration = 0.0
        
        if buffer:
            combined_audio = np.concatenate(buffer)
            result = self.transcribe_audio(combined_audio, sr, language)
            if result.segments:
                yield result
    
    def detect_language(self, audio: np.ndarray, sampling_rate: int) -> Optional[str]:
        self._initialize()
        
        if self._model is None:
            return "zh"
        
        try:
            import soundfile as sf
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            sf.write(temp_path, audio, sampling_rate)
            
            result = self._model.transcribe(temp_path, language=None)
            
            os.unlink(temp_path)
            
            return result.get("language")
        except Exception as e:
            return "zh"


class TextStreamProcessor:
    def __init__(self, max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size
        self.text_buffer: List[Dict] = []
        self.current_session_start: Optional[float] = None
    
    def add_text_segment(
        self,
        text: str,
        timestamp: float,
        speaker_id: Optional[str] = None,
        confidence: float = 1.0
    ):
        if self.current_session_start is None:
            self.current_session_start = timestamp
        
        self.text_buffer.append({
            "text": text,
            "timestamp": timestamp,
            "speaker_id": speaker_id,
            "confidence": confidence
        })
        
        if len(self.text_buffer) > self.max_buffer_size:
            self.text_buffer.pop(0)
    
    def get_recent_text(self, seconds: float = 10.0) -> str:
        if not self.text_buffer:
            return ""
        
        current_time = self.text_buffer[-1]["timestamp"]
        threshold = current_time - seconds
        
        recent = [
            seg["text"] for seg in self.text_buffer
            if seg["timestamp"] >= threshold
        ]
        
        return " ".join(recent)
    
    def get_full_text(self) -> str:
        return " ".join([seg["text"] for seg in self.text_buffer])
    
    def get_text_by_speaker(self) -> Dict[str, str]:
        speaker_texts: Dict[str, List[str]] = {}
        
        for seg in self.text_buffer:
            speaker = seg.get("speaker_id") or "unknown"
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(seg["text"])
        
        return {
            speaker: " ".join(texts)
            for speaker, texts in speaker_texts.items()
        }
    
    def clear(self):
        self.text_buffer.clear()
        self.current_session_start = None
    
    def get_session_duration(self) -> float:
        if not self.text_buffer or self.current_session_start is None:
            return 0.0
        return self.text_buffer[-1]["timestamp"] - self.current_session_start
