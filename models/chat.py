from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    text: Optional[str] = None
    images: Optional[List[str]] = None

class StoredMessage(BaseModel):
    user_email: str
    role: str
    text: Optional[str] = None
    images: Optional[List[str]] = None
