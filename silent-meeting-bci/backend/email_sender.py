import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import List, Optional, Dict
from dataclasses import dataclass
import ssl


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    use_ssl: bool = True
    use_tls: bool = True


@dataclass
class EmailAttachment:
    filename: str
    content: bytes
    mime_type: str = "application/octet-stream"


class EmailSender:
    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config or self._load_config_from_env()
        self._connection = None
    
    def _load_config_from_env(self) -> EmailConfig:
        return EmailConfig(
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            username=os.getenv("SMTP_USERNAME", ""),
            password=os.getenv("SMTP_PASSWORD", ""),
            from_email=os.getenv("SMTP_FROM_EMAIL", os.getenv("SMTP_USERNAME", "")),
            use_ssl=os.getenv("SMTP_USE_SSL", "true").lower() == "true",
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        )
    
    def _connect(self):
        if self._connection:
            return
        
        if self.config.use_ssl:
            context = ssl.create_default_context()
            self._connection = smtplib.SMTP_SSL(
                self.config.smtp_host,
                self.config.smtp_port,
                context=context
            )
        else:
            self._connection = smtplib.SMTP(
                self.config.smtp_host,
                self.config.smtp_port
            )
            
            if self.config.use_tls:
                context = ssl.create_default_context()
                self._connection.starttls(context=context)
        
        if self.config.username and self.config.password:
            self._connection.login(self.config.username, self.config.password)
    
    def _disconnect(self):
        if self._connection:
            try:
                self._connection.quit()
            except Exception:
                pass
            self._connection = None
    
    def send_email(
        self,
        to_emails: List[str],
        subject: str,
        body: str,
        cc_emails: Optional[List[str]] = None,
        bcc_emails: Optional[List[str]] = None,
        attachments: Optional[List[EmailAttachment]] = None,
        is_html: bool = False
    ) -> bool:
        try:
            self._connect()
            
            msg = MIMEMultipart()
            msg["From"] = formataddr(("无声会议纪要系统", self.config.from_email))
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = subject
            
            if cc_emails:
                msg["Cc"] = ", ".join(cc_emails)
            
            if is_html:
                msg.attach(MIMEText(body, "html", "utf-8"))
            else:
                msg.attach(MIMEText(body, "plain", "utf-8"))
            
            if attachments:
                for attachment in attachments:
                    part = MIMEBase(*attachment.mime_type.split("/"))
                    part.set_payload(attachment.content)
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {attachment.filename}"
                    )
                    msg.attach(part)
            
            all_recipients = to_emails + (cc_emails or []) + (bcc_emails or [])
            
            self._connection.sendmail(
                self.config.from_email,
                all_recipients,
                msg.as_string()
            )
            
            return True
        except Exception as e:
            print(f"发送邮件失败: {e}")
            return False
        finally:
            self._disconnect()
    
    def send_meeting_minutes(
        self,
        to_emails: List[str],
        meeting_title: str,
        meeting_date: str,
        markdown_content: str,
        markdown_filename: str,
        participants: Optional[List[str]] = None,
        cc_emails: Optional[List[str]] = None,
        additional_notes: Optional[str] = None
    ) -> bool:
        subject = f"【会议纪要】{meeting_title} - {meeting_date}"
        
        body = self._build_email_body(
            meeting_title,
            meeting_date,
            participants,
            additional_notes
        )
        
        attachment = EmailAttachment(
            filename=markdown_filename,
            content=markdown_content.encode("utf-8"),
            mime_type="text/markdown"
        )
        
        return self.send_email(
            to_emails=to_emails,
            subject=subject,
            body=body,
            cc_emails=cc_emails,
            attachments=[attachment],
            is_html=True
        )
    
    def _build_email_body(
        self,
        meeting_title: str,
        meeting_date: str,
        participants: Optional[List[str]],
        additional_notes: Optional[str]
    ) -> str:
        participants_str = ""
        if participants:
            participants_str = f"<p><strong>参会人员：</strong>{', '.join(participants)}</p>"
        
        notes_str = ""
        if additional_notes:
            notes_str = f"<p><strong>备注：</strong>{additional_notes}</p>"
        
        html_body = f"""
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 20px;
        }}
        .content {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .footer {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            font-size: 12px;
            color: #666;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📋 无声会议纪要</h1>
    </div>
    
    <div class="content">
        <p><strong>会议主题：</strong>{meeting_title}</p>
        <p><strong>会议日期：</strong>{meeting_date}</p>
        {participants_str}
        
        <div class="highlight">
            <p>您好！本次无声会议的纪要已生成，请查收附件中的详细内容。</p>
            <p>纪要包含：关键要点、会议决定、行动项、讨论主题等内容。</p>
        </div>
        
        {notes_str}
        
        <p>如有任何问题，请随时联系。</p>
    </div>
    
    <div class="footer">
        <p>此邮件由无声会议思想纪要系统自动发送，请勿直接回复。</p>
        <p>本系统通过脑机接口技术捕捉思维信号，为闭锁综合征患者提供参会新方式。</p>
    </div>
</body>
</html>
"""
        
        return html_body
    
    def test_connection(self) -> bool:
        try:
            self._connect()
            return True
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False
        finally:
            self._disconnect()
