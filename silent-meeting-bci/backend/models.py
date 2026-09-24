from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SignalSource(str, Enum):
    BCI = "bci"
    AUDIO = "audio"
    FUSED = "fused"


class BCIConnectionStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class MeetingStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"


class BCIDataRequest(BaseModel):
    eeg_data: List[List[float]] = Field(..., description="EEG通道数据 [channels x samples]")
    sampling_rate: int = Field(250, description="采样率")
    timestamp: Optional[float] = None
    session_id: Optional[str] = None


class AudioDataRequest(BaseModel):
    audio_data: List[float] = Field(..., description="音频数据")
    sampling_rate: int = Field(16000, description="采样率")
    timestamp: Optional[float] = None
    session_id: Optional[str] = None


class DecodedWordResponse(BaseModel):
    word: str
    confidence: float
    timestamp: float
    speaker_id: Optional[str] = None
    source: SignalSource


class TextSegmentResponse(BaseModel):
    text: str
    confidence: float
    timestamp: float
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    source: SignalSource


class MeetingParticipant(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    bci_connected: bool = False
    signal_quality: float = 0.0


class MeetingCreateRequest(BaseModel):
    title: str
    participants: List[MeetingParticipant]
    description: Optional[str] = None
    start_time: Optional[datetime] = None


class MeetingUpdateRequest(BaseModel):
    status: Optional[MeetingStatus] = None
    participants: Optional[List[MeetingParticipant]] = None
    end_time: Optional[datetime] = None


class ActionItem(BaseModel):
    task: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[str] = "medium"


class DiscussionTopic(BaseModel):
    topic: str
    summary: Optional[str] = None


class MeetingSummaryResponse(BaseModel):
    title: str
    date: str
    participants: List[str]
    agenda: str
    key_points: List[str]
    decisions: List[str]
    action_items: List[ActionItem]
    discussion_topics: List[DiscussionTopic]
    next_meeting: Optional[str] = None
    additional_notes: Optional[str] = None
    markdown_content: Optional[str] = None


class SignalQualityResponse(BaseModel):
    overall_quality: float
    channel_qualities: List[float]
    artifacts: List[str]
    timestamp: float


class EmailSendRequest(BaseModel):
    to_emails: List[str]
    cc_emails: Optional[List[str]] = None
    subject: Optional[str] = None
    meeting_title: str
    meeting_date: str
    participants: Optional[List[str]] = None
    additional_notes: Optional[str] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    status: MeetingStatus
    participants: List[MeetingParticipant]
    duration_seconds: float
    word_count: int
    decoded_words: List[DecodedWordResponse]
    start_time: Optional[datetime] = None


class HealthCheckResponse(BaseModel):
    status: str
    bci_connected: bool
    audio_ready: bool
    whisper_available: bool
    openai_available: bool
    timestamp: datetime
