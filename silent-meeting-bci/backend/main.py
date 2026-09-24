import os
import uuid
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
import numpy as np

from .bci_processor import SilentSpeechDecoder, DecodedWord
from .audio_processor import AudioProcessor
from .speaker_diarization import EEGSpeakerDiarizer, AudioSpeakerDiarizer
from .signal_fusion import SignalFusion, TextSegment, SignalSource
from .whisper_processor import WhisperProcessor, TextStreamProcessor
from .summary_generator import OpenAISummarizer, ThoughtOrganizer, MeetingSummary, SummaryConfig
from .markdown_generator import MarkdownGenerator
from .email_sender import EmailSender

from .models import (
    BCIDataRequest,
    AudioDataRequest,
    DecodedWordResponse,
    TextSegmentResponse,
    MeetingCreateRequest,
    MeetingUpdateRequest,
    MeetingSummaryResponse,
    MeetingParticipant,
    SignalQualityResponse,
    EmailSendRequest,
    SessionStatusResponse,
    HealthCheckResponse,
    MeetingStatus,
    ActionItem,
    DiscussionTopic,
)

load_dotenv()

app = FastAPI(
    title="无声会议思想纪要系统 API",
    description="结合脑机接口的无声会议思想纪要全栈系统",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MeetingSession:
    def __init__(self, session_id: str, title: str, participants: List[MeetingParticipant]):
        self.session_id = session_id
        self.title = title
        self.participants = participants
        self.status = MeetingStatus.NOT_STARTED
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        self.decoded_words: List[DecodedWord] = []
        self.text_segments: List[TextSegment] = []
        
        self.bci_decoder = SilentSpeechDecoder()
        self.audio_processor = AudioProcessor()
        self.eeg_diarizer = EEGSpeakerDiarizer()
        self.signal_fusion = SignalFusion()
        self.whisper_processor = WhisperProcessor(language="zh")
        self.text_processor = TextStreamProcessor()
        self.thought_organizer = ThoughtOrganizer()
        
        self.participant_buffers: Dict[str, List[DecodedWord]] = defaultdict(list)


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, MeetingSession] = {}
        self.active_connections: Dict[str, List[WebSocket]] = defaultdict(list)
    
    def create_session(self, title: str, participants: List[MeetingParticipant]) -> MeetingSession:
        session_id = str(uuid.uuid4())
        session = MeetingSession(session_id, title, participants)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[MeetingSession]:
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    async def broadcast_to_session(self, session_id: str, message: Dict):
        if session_id in self.active_connections:
            for ws in self.active_connections[session_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass


session_manager = SessionManager()
openai_summarizer = OpenAISummarizer(config=SummaryConfig(language="zh"))
markdown_generator = MarkdownGenerator()
email_sender = EmailSender()


@app.get("/")
async def root():
    return {
        "name": "无声会议思想纪要系统",
        "version": "1.0.0",
        "description": "结合脑机接口的无声会议思想纪要全栈系统，为闭锁综合征患者提供参会新方式"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        bci_connected=True,
        audio_ready=True,
        whisper_available=openai_summarizer._client is not None,
        openai_available=openai_summarizer.api_key is not None,
        timestamp=datetime.now()
    )


@app.post("/api/sessions")
async def create_session(request: MeetingCreateRequest):
    session = session_manager.create_session(request.title, request.participants)
    session.status = MeetingStatus.IN_PROGRESS
    session.start_time = request.start_time or datetime.now()
    
    return {
        "session_id": session.session_id,
        "title": session.title,
        "status": session.status,
        "start_time": session.start_time
    }


@app.get("/api/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    duration = 0.0
    if session.start_time:
        duration = (datetime.now() - session.start_time).total_seconds()
    
    decoded_words_response = [
        DecodedWordResponse(
            word=w.word,
            confidence=w.confidence,
            timestamp=w.timestamp,
            speaker_id=w.speaker_id,
            source=SignalSource.BCI
        )
        for w in session.decoded_words[-20:]
    ]
    
    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
        participants=session.participants,
        duration_seconds=duration,
        word_count=len(session.decoded_words),
        decoded_words=decoded_words_response,
        start_time=session.start_time
    )


@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, request: MeetingUpdateRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if request.status:
        session.status = request.status
        if request.status == MeetingStatus.COMPLETED:
            session.end_time = datetime.now()
    
    if request.participants:
        session.participants = request.participants
    
    if request.end_time:
        session.end_time = request.end_time
    
    return {"status": "success", "session_id": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"status": "success"}


@app.post("/api/sessions/{session_id}/bci-data")
async def process_bci_data(session_id: str, request: BCIDataRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    eeg_array = np.array(request.eeg_data)
    timestamp = request.timestamp or time.time()
    
    decoded_word = session.bci_decoder.process_eeg_packet(eeg_array)
    
    if decoded_word:
        decoded_word.timestamp = timestamp
        session.decoded_words.append(decoded_word)
        
        bci_segment = TextSegment(
            text=decoded_word.word,
            confidence=decoded_word.confidence,
            timestamp=timestamp,
            source=SignalSource.BCI,
            speaker_id=decoded_word.speaker_id,
            start_time=timestamp,
            end_time=timestamp
        )
        session.signal_fusion.add_bci_segment(bci_segment)
        session.thought_organizer.add_fragment(
            text=decoded_word.word,
            timestamp=timestamp,
            speaker_id=decoded_word.speaker_id,
            confidence=decoded_word.confidence,
            source="bci"
        )
        
        await session_manager.broadcast_to_session(session_id, {
            "type": "decoded_word",
            "data": {
                "word": decoded_word.word,
                "confidence": decoded_word.confidence,
                "timestamp": timestamp,
                "speaker_id": decoded_word.speaker_id
            }
        })
        
        return {
            "status": "success",
            "decoded": True,
            "word": decoded_word.word,
            "confidence": decoded_word.confidence
        }
    
    return {"status": "success", "decoded": False}


@app.post("/api/sessions/{session_id}/audio-data")
async def process_audio_data(session_id: str, request: AudioDataRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    audio_array = np.array(request.audio_data)
    timestamp = request.timestamp or time.time()
    
    processed_audio = session.audio_processor.preprocess(audio_array)
    enhanced_audio = session.audio_processor.enhance_weak_audio(processed_audio)
    
    transcription = session.whisper_processor.transcribe_audio(
        enhanced_audio,
        request.sampling_rate
    )
    
    for seg in transcription.segments:
        audio_segment = TextSegment(
            text=seg.text,
            confidence=seg.confidence,
            timestamp=timestamp + seg.start_time,
            source=SignalSource.AUDIO,
            speaker_id=seg.speaker_id,
            start_time=timestamp + seg.start_time,
            end_time=timestamp + seg.end_time
        )
        session.signal_fusion.add_audio_segment(audio_segment)
        session.text_segments.append(audio_segment)
        session.thought_organizer.add_fragment(
            text=seg.text,
            timestamp=timestamp + seg.start_time,
            speaker_id=seg.speaker_id,
            confidence=seg.confidence,
            source="audio"
        )
        
        await session_manager.broadcast_to_session(session_id, {
            "type": "audio_transcript",
            "data": {
                "text": seg.text,
                "confidence": seg.confidence,
                "timestamp": timestamp + seg.start_time,
                "speaker_id": seg.speaker_id
            }
        })
    
    return {
        "status": "success",
        "transcript": transcription.full_text,
        "segments": len(transcription.segments)
    }


@app.post("/api/sessions/{session_id}/audio-upload")
async def upload_audio_file(session_id: str, file: UploadFile = File(...)):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        transcription = session.whisper_processor.transcribe_file(temp_file.name)
        
        timestamp = time.time()
        for seg in transcription.segments:
            audio_segment = TextSegment(
                text=seg.text,
                confidence=seg.confidence,
                timestamp=timestamp + seg.start_time,
                source=SignalSource.AUDIO,
                speaker_id=seg.speaker_id,
                start_time=timestamp + seg.start_time,
                end_time=timestamp + seg.end_time
            )
            session.text_segments.append(audio_segment)
            session.thought_organizer.add_fragment(
                text=seg.text,
                timestamp=timestamp + seg.start_time,
                speaker_id=seg.speaker_id,
                confidence=seg.confidence,
                source="audio_file"
            )
        
        return {
            "status": "success",
            "filename": file.filename,
            "transcript": transcription.full_text,
            "segments": len(transcription.segments)
        }
    finally:
        import os
        os.unlink(temp_file.name)


@app.get("/api/sessions/{session_id}/signal-quality", response_model=SignalQualityResponse)
async def get_signal_quality(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.decoded_words:
        return SignalQualityResponse(
            overall_quality=0.0,
            channel_qualities=[],
            artifacts=[],
            timestamp=time.time()
        )
    
    recent_confidences = [w.confidence for w in session.decoded_words[-10:]]
    overall_quality = np.mean(recent_confidences) if recent_confidences else 0.0
    
    return SignalQualityResponse(
        overall_quality=overall_quality,
        channel_qualities=[overall_quality] * 8,
        artifacts=[],
        timestamp=time.time()
    )


@app.get("/api/sessions/{session_id}/fuse")
async def fuse_signals(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    fusion_result = session.signal_fusion.fuse()
    
    segments_response = [
        TextSegmentResponse(
            text=seg.text,
            confidence=seg.confidence,
            timestamp=seg.timestamp,
            start_time=seg.start_time or seg.timestamp,
            end_time=seg.end_time or seg.timestamp,
            speaker_id=seg.speaker_id,
            source=seg.source
        )
        for seg in fusion_result.segments
    ]
    
    return {
        "status": "success",
        "fused_text": fusion_result.fused_text,
        "overall_confidence": fusion_result.overall_confidence,
        "segments": segments_response
    }


@app.get("/api/sessions/{session_id}/summary", response_model=MeetingSummaryResponse)
async def generate_summary(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    organized_thoughts = session.thought_organizer.organize()
    coherent_text = session.thought_organizer.get_coherent_text()
    
    participant_names = [p.name for p in session.participants]
    
    summary = openai_summarizer.generate_summary(
        meeting_text=coherent_text,
        participants=participant_names,
        meeting_title=session.title
    )
    
    action_items = [
        ActionItem(
            task=item.get("task", ""),
            assignee=item.get("assignee"),
            deadline=item.get("deadline"),
            priority=item.get("priority", "medium")
        )
        for item in summary.action_items
    ]
    
    discussion_topics = [
        DiscussionTopic(
            topic=topic.get("topic", ""),
            summary=topic.get("summary")
        )
        for topic in summary.discussion_topics
    ]
    
    markdown_content = markdown_generator.generate_markdown(summary)
    
    return MeetingSummaryResponse(
        title=summary.title,
        date=summary.date,
        participants=summary.participants,
        agenda=summary.agenda,
        key_points=summary.key_points,
        decisions=summary.decisions,
        action_items=action_items,
        discussion_topics=discussion_topics,
        next_meeting=summary.next_meeting,
        additional_notes=summary.additional_notes,
        markdown_content=markdown_content
    )


@app.get("/api/sessions/{session_id}/markdown")
async def get_markdown(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    organized_thoughts = session.thought_organizer.organize()
    coherent_text = session.thought_organizer.get_coherent_text()
    
    participant_names = [p.name for p in session.participants]
    
    summary = openai_summarizer.generate_summary(
        meeting_text=coherent_text,
        participants=participant_names,
        meeting_title=session.title
    )
    
    speaker_segments = [
        {
            "speaker_id": t.get("speaker_id"),
            "text": t.get("text"),
            "start_time": t.get("timestamp")
        }
        for t in organized_thoughts
    ]
    
    markdown_content = markdown_generator.generate_detailed_markdown(
        summary,
        raw_transcript=coherent_text,
        speaker_segments=speaker_segments
    )
    
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
    temp_file.write(markdown_content)
    temp_file.close()
    
    filename = markdown_generator.generate_filename(session.title, summary.date)
    
    return FileResponse(
        temp_file.name,
        media_type="text/markdown",
        filename=filename
    )


@app.post("/api/sessions/{session_id}/send-email")
async def send_summary_email(session_id: str, request: EmailSendRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    organized_thoughts = session.thought_organizer.organize()
    coherent_text = session.thought_organizer.get_coherent_text()
    
    participant_names = [p.name for p in session.participants]
    
    summary = openai_summarizer.generate_summary(
        meeting_text=coherent_text,
        participants=participant_names,
        meeting_title=session.title
    )
    
    markdown_content = markdown_generator.generate_markdown(summary)
    filename = markdown_generator.generate_filename(session.title, summary.date)
    
    success = email_sender.send_meeting_minutes(
        to_emails=request.to_emails,
        meeting_title=request.meeting_title,
        meeting_date=request.meeting_date,
        markdown_content=markdown_content,
        markdown_filename=filename,
        participants=request.participants or participant_names,
        cc_emails=request.cc_emails,
        additional_notes=request.additional_notes
    )
    
    if success:
        return {"status": "success", "message": "邮件发送成功"}
    else:
        raise HTTPException(status_code=500, detail="邮件发送失败")


@app.get("/api/sessions")
async def list_sessions():
    sessions_list = []
    for session_id, session in session_manager.sessions.items():
        duration = 0.0
        if session.start_time:
            duration = (datetime.now() - session.start_time).total_seconds()
        
        sessions_list.append({
            "session_id": session_id,
            "title": session.title,
            "status": session.status,
            "participant_count": len(session.participants),
            "word_count": len(session.decoded_words),
            "duration_seconds": duration,
            "start_time": session.start_time
        })
    
    return {"sessions": sessions_list}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    session_manager.active_connections[session_id].append(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "bci_data":
                eeg_data = np.array(data.get("eeg_data", []))
                timestamp = data.get("timestamp", time.time())
                
                decoded_word = session.bci_decoder.process_eeg_packet(eeg_data)
                
                if decoded_word:
                    decoded_word.timestamp = timestamp
                    session.decoded_words.append(decoded_word)
                    
                    await websocket.send_json({
                        "type": "decoded_word",
                        "data": {
                            "word": decoded_word.word,
                            "confidence": decoded_word.confidence,
                            "timestamp": timestamp,
                            "speaker_id": decoded_word.speaker_id
                        }
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
    
    except WebSocketDisconnect:
        session_manager.active_connections[session_id].remove(websocket)
    except Exception as e:
        if websocket in session_manager.active_connections[session_id]:
            session_manager.active_connections[session_id].remove(websocket)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
