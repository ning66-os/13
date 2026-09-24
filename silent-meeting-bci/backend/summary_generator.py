import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class MeetingSummary:
    title: str
    date: str
    participants: List[str]
    agenda: str
    key_points: List[str]
    decisions: List[str]
    action_items: List[Dict]
    discussion_topics: List[Dict]
    next_meeting: Optional[str] = None
    additional_notes: Optional[str] = None


@dataclass
class SummaryConfig:
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    language: str = "zh"
    include_timestamps: bool = True
    include_speaker_labels: bool = True


class OpenAISummarizer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[SummaryConfig] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = config or SummaryConfig()
        self._client = None
        
        if self.api_key:
            self._init_client()
    
    def _init_client(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            self._client = None
    
    def generate_summary(
        self,
        meeting_text: str,
        participants: Optional[List[str]] = None,
        meeting_title: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> MeetingSummary:
        if self._client is not None:
            return self._generate_with_openai(
                meeting_text, participants, meeting_title, additional_context
            )
        else:
            return self._generate_simulation(
                meeting_text, participants, meeting_title, additional_context
            )
    
    def _generate_with_openai(
        self,
        meeting_text: str,
        participants: Optional[List[str]],
        meeting_title: Optional[str],
        additional_context: Optional[str]
    ) -> MeetingSummary:
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                meeting_text, participants, meeting_title, additional_context
            )
            
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            summary_text = response.choices[0].message.content
            return self._parse_summary_response(summary_text)
        except Exception as e:
            return self._generate_simulation(
                meeting_text, participants, meeting_title, additional_context
            )
    
    def _build_system_prompt(self) -> str:
        lang_prompt = "中文" if self.config.language == "zh" else "English"
        
        return f"""你是一个专业的会议纪要助手。请根据提供的会议文本内容，生成一份结构清晰、专业的会议摘要。

要求：
1. 识别会议的主要议题和讨论内容
2. 提取关键决策和行动项
3. 总结讨论的要点
4. 用{lang_prompt}输出
5. 格式化为JSON结构，包含以下字段：
   - title: 会议标题
   - date: 会议日期
   - participants: 参会人员列表
   - agenda: 会议议程简述
   - key_points: 关键要点列表
   - decisions: 做出的决定列表
   - action_items: 行动项列表，每个包含task、assignee、deadline字段
   - discussion_topics: 讨论主题列表，每个包含topic、summary字段
   - next_meeting: 下次会议时间（如未提及则为null）
   - additional_notes: 其他备注

请确保摘要准确、简洁，涵盖所有重要信息。"""
    
    def _build_user_prompt(
        self,
        meeting_text: str,
        participants: Optional[List[str]],
        meeting_title: Optional[str],
        additional_context: Optional[str]
    ) -> str:
        prompt_parts = []
        
        if meeting_title:
            prompt_parts.append(f"会议标题: {meeting_title}")
        
        if participants:
            prompt_parts.append(f"参会人员: {', '.join(participants)}")
        
        if additional_context:
            prompt_parts.append(f"背景信息: {additional_context}")
        
        prompt_parts.append(f"\n会议内容:\n{meeting_text}")
        prompt_parts.append("\n请生成会议摘要JSON。")
        
        return "\n".join(prompt_parts)
    
    def _parse_summary_response(self, response_text: str) -> MeetingSummary:
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return MeetingSummary(
                    title=data.get("title", "会议摘要"),
                    date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
                    participants=data.get("participants", []),
                    agenda=data.get("agenda", ""),
                    key_points=data.get("key_points", []),
                    decisions=data.get("decisions", []),
                    action_items=data.get("action_items", []),
                    discussion_topics=data.get("discussion_topics", []),
                    next_meeting=data.get("next_meeting"),
                    additional_notes=data.get("additional_notes")
                )
        except Exception as e:
            pass
        
        return self._parse_text_summary(response_text)
    
    def _parse_text_summary(self, text: str) -> MeetingSummary:
        lines = text.strip().split('\n')
        
        key_points = []
        decisions = []
        action_items = []
        discussion_topics = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if '要点' in line or '关键点' in line:
                current_section = 'key_points'
            elif '决定' in line or '决策' in line:
                current_section = 'decisions'
            elif '行动' in line or '待办' in line:
                current_section = 'action_items'
            elif '讨论' in line or '议题' in line:
                current_section = 'discussion_topics'
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                item = line[1:].strip()
                if current_section == 'key_points':
                    key_points.append(item)
                elif current_section == 'decisions':
                    decisions.append(item)
                elif current_section == 'action_items':
                    action_items.append({"task": item, "assignee": "", "deadline": ""})
                elif current_section == 'discussion_topics':
                    discussion_topics.append({"topic": item, "summary": ""})
        
        return MeetingSummary(
            title="会议摘要",
            date=datetime.now().strftime("%Y-%m-%d"),
            participants=[],
            agenda="",
            key_points=key_points[:10],
            decisions=decisions[:10],
            action_items=action_items[:10],
            discussion_topics=discussion_topics[:10]
        )
    
    def _generate_simulation(
        self,
        meeting_text: str,
        participants: Optional[List[str]],
        meeting_title: Optional[str],
        additional_context: Optional[str]
    ) -> MeetingSummary:
        sentences = meeting_text.split('。')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        key_points = sentences[:min(5, len(sentences))]
        decisions = sentences[min(2, len(sentences)):min(7, len(sentences))]
        
        action_items = []
        for i, sent in enumerate(sentences[:3]):
            action_items.append({
                "task": sent,
                "assignee": f"参会者{i % 3 + 1}",
                "deadline": "下周"
            })
        
        discussion_topics = []
        topics = ["项目进展", "技术方案", "资源调配", "风险管理"]
        for i, topic in enumerate(topics[:3]):
            discussion_topics.append({
                "topic": topic,
                "summary": sentences[i] if i < len(sentences) else ""
            })
        
        return MeetingSummary(
            title=meeting_title or "无声会议思想纪要",
            date=datetime.now().strftime("%Y-%m-%d"),
            participants=participants or ["参会者1", "参会者2", "参会者3"],
            agenda="讨论项目进展和下一步计划",
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            discussion_topics=discussion_topics,
            next_meeting="待定",
            additional_notes=additional_context
        )
    
    def summarize_segments(
        self,
        segments: List[Dict],
        participants: Optional[List[str]] = None
    ) -> MeetingSummary:
        full_text = ""
        
        for seg in segments:
            text = seg.get("text", "")
            speaker = seg.get("speaker_id", "")
            
            if self.config.include_speaker_labels and speaker:
                full_text += f"[{speaker}] "
            
            full_text += text
            
            if self.config.include_timestamps:
                start = seg.get("start_time", 0)
                full_text += f" ({start:.1f}s)"
            
            full_text += "\n"
        
        return self.generate_summary(full_text, participants)
    
    def extract_action_items(self, text: str) -> List[Dict]:
        if self._client is None:
            return []
        
        try:
            prompt = f"""请从以下文本中提取所有行动项（待办事项）。

文本：
{text}

请以JSON格式输出，格式如下：
{{
  "action_items": [
    {{
      "task": "任务描述",
      "assignee": "负责人",
      "deadline": "截止时间",
      "priority": "优先级"
    }}
  ]
}}

如果没有提到负责人或截止时间，请留空字符串。"""
            
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                data = json.loads(result_text[json_start:json_end])
                return data.get("action_items", [])
        except Exception as e:
            pass
        
        return []


class ThoughtOrganizer:
    def __init__(self):
        self.thought_fragments: List[Dict] = []
    
    def add_fragment(
        self,
        text: str,
        timestamp: float,
        speaker_id: Optional[str] = None,
        confidence: float = 1.0,
        source: Optional[str] = None
    ):
        self.thought_fragments.append({
            "text": text,
            "timestamp": timestamp,
            "speaker_id": speaker_id,
            "confidence": confidence,
            "source": source
        })
    
    def organize(self) -> List[Dict]:
        if not self.thought_fragments:
            return []
        
        sorted_fragments = sorted(self.thought_fragments, key=lambda x: x["timestamp"])
        
        grouped = self._group_by_speaker(sorted_fragments)
        
        merged = self._merge_adjacent_fragments(grouped)
        
        return merged
    
    def _group_by_speaker(self, fragments: List[Dict]) -> List[Dict]:
        speakers: Dict[str, List[Dict]] = {}
        
        for frag in fragments:
            speaker = frag.get("speaker_id") or "unknown"
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(frag)
        
        result = []
        for speaker, frags in speakers.items():
            result.extend(sorted(frags, key=lambda x: x["timestamp"]))
        
        return sorted(result, key=lambda x: x["timestamp"])
    
    def _merge_adjacent_fragments(self, fragments: List[Dict]) -> List[Dict]:
        if not fragments:
            return []
        
        merged = [fragments[0].copy()]
        
        for frag in fragments[1:]:
            last = merged[-1]
            
            time_diff = frag["timestamp"] - last["timestamp"]
            same_speaker = frag.get("speaker_id") == last.get("speaker_id")
            
            if time_diff < 5.0 and same_speaker:
                last["text"] = f"{last['text']} {frag['text']}"
                last["confidence"] = (last["confidence"] + frag["confidence"]) / 2
                last["end_timestamp"] = frag["timestamp"]
            else:
                new_frag = frag.copy()
                new_frag["end_timestamp"] = frag["timestamp"]
                merged.append(new_frag)
        
        for frag in merged:
            if "end_timestamp" not in frag:
                frag["end_timestamp"] = frag["timestamp"]
        
        return merged
    
    def get_coherent_text(self) -> str:
        organized = self.organize()
        return " ".join([frag["text"] for frag in organized])
    
    def clear(self):
        self.thought_fragments.clear()
