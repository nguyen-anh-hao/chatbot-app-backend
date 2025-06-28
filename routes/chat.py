from fastapi import APIRouter, Request
from models.chat import ChatMessage
from database.mongodb import chat_collection

chat_router = APIRouter()

@chat_router.post("/api/chat")
async def chat(msg: ChatMessage, request: Request):
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}

    # Lưu tin nhắn người dùng
    await chat_collection.insert_one({
        "user_email": user["email"],
        "role": "user",
        "text": msg.text,
        "images": msg.images,
    })

    bot_reply = "Xin chào, tôi đã nhận được tin nhắn!"
    
    # Lưu phản hồi bot
    await chat_collection.insert_one({
        "user_email": user["email"],
        "role": "bot",
        "text": bot_reply,
        "images": [],
    })

    return {"reply": bot_reply}
