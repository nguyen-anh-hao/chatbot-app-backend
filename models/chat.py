from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    text: Optional[str] = None
    images: Optional[List[str]] = None
    conversation_id: Optional[str] = None

class StoredMessage(BaseModel):
    user_email: str
    conversation_id: str
    role: str
    text: Optional[str] = None
    images: Optional[List[str]] = None
    timestamp: datetime

class Conversation(BaseModel):
    user_email: str
    topic: str
    created_at: datetime
    last_message_at: datetime
    message_count: int = 0
