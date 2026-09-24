from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import os


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


class MarkdownGenerator:
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path
        self._default_template = self._get_default_template()
    
    def _get_default_template(self) -> str:
        return """# {title}

**日期**: {date}
**参会人员**: {participants}

---

## 会议议程

{agenda}

---

## 关键要点

{key_points}

---

## 会议决定

{decisions}

---

## 行动项

| 任务 | 负责人 | 截止时间 | 优先级 |
|------|--------|----------|--------|
{action_items_table}

---

## 讨论主题

{discussion_topics}

---

## 下次会议

{next_meeting}

---

## 备注

{additional_notes}

---

*本纪要由无声会议思想纪要系统自动生成*
"""
    
    def generate_markdown(self, summary: MeetingSummary) -> str:
        participants_str = ", ".join(summary.participants) if summary.participants else "未记录"
        
        key_points_str = self._format_list(summary.key_points)
        decisions_str = self._format_list(summary.decisions)
        
        action_items_table = self._format_action_items_table(summary.action_items)
        discussion_topics_str = self._format_discussion_topics(summary.discussion_topics)
        
        next_meeting_str = summary.next_meeting or "待定"
        additional_notes_str = summary.additional_notes or "无"
        
        markdown = self._default_template.format(
            title=summary.title,
            date=summary.date,
            participants=participants_str,
            agenda=summary.agenda or "未记录",
            key_points=key_points_str,
            decisions=decisions_str,
            action_items_table=action_items_table,
            discussion_topics=discussion_topics_str,
            next_meeting=next_meeting_str,
            additional_notes=additional_notes_str
        )
        
        return markdown
    
    def _format_list(self, items: List[str]) -> str:
        if not items:
            return "- 无"
        return "\n".join([f"- {item}" for item in items])
    
    def _format_action_items_table(self, action_items: List[Dict]) -> str:
        if not action_items:
            return "| 无 | - | - | - |"
        
        rows = []
        for item in action_items:
            task = item.get("task", "")
            assignee = item.get("assignee", "-")
            deadline = item.get("deadline", "-")
            priority = item.get("priority", "-")
            
            row = f"| {task} | {assignee} | {deadline} | {priority} |"
            rows.append(row)
        
        return "\n".join(rows)
    
    def _format_discussion_topics(self, topics: List[Dict]) -> str:
        if not topics:
            return "### 无\n\n暂无讨论主题记录。"
        
        sections = []
        for topic in topics:
            title = topic.get("topic", "未命名主题")
            summary = topic.get("summary", "无详细内容")
            sections.append(f"### {title}\n\n{summary}\n")
        
        return "\n".join(sections)
    
    def generate_detailed_markdown(
        self,
        summary: MeetingSummary,
        raw_transcript: Optional[str] = None,
        speaker_segments: Optional[List[Dict]] = None
    ) -> str:
        basic_md = self.generate_markdown(summary)
        
        additional_sections = []
        
        if raw_transcript:
            additional_sections.append(f"""
---

## 完整转录

```
{raw_transcript}
```
""")
        
        if speaker_segments:
            speaker_sections = []
            for seg in speaker_segments:
                speaker = seg.get("speaker_id", "未知")
                text = seg.get("text", "")
                timestamp = seg.get("start_time", 0)
                
                speaker_sections.append(f"**[{timestamp:.1f}s] {speaker}**: {text}")
            
            additional_sections.append(f"""
---

## 发言记录

{chr(10).join(speaker_sections)}
""")
        
        return basic_md + "\n".join(additional_sections)
    
    def save_markdown(self, markdown: str, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        return output_path
    
    def generate_filename(self, title: str, date: Optional[str] = None) -> str:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        return f"{date}_{safe_title}.md"
