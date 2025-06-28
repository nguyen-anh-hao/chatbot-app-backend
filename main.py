from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

import base64

app = FastAPI()




origins = [
    "http://localhost:5173",  # Frontend của bạn
    # Có thể thêm nhiều origin nếu cần
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Cho phép từ frontend
    allow_credentials=True,
    allow_methods=["*"],    # Cho phép mọi method GET, POST,...
    allow_headers=["*"],    # Cho phép mọi header
)

class ChatMessage(BaseModel):
    text: Optional[str] = None
    images: Optional[List[str]] = None  # Danh sách base64 ảnh

@app.post("/api/chat")
async def chat(msg: ChatMessage):
    if msg.text:
        print(f"Received text message: {msg.text}")
    if msg.images:
        print(f"Received {len(msg.images)} images:")
        for i, img_b64 in enumerate(msg.images):
            print(f" Image {i+1} starts with: {img_b64[:30]}...")  # In phần đầu base64
    # Trả về response dummy
    return {"reply": "Backend received your message."}
