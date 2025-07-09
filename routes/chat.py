from fastapi import APIRouter, Request
from models.chat import ChatMessage
from database.mongodb import chat_collection, conversation_collection
from datetime import datetime
from bson import ObjectId
import re
from llama_production import LlamaChatModel  # ✅ Dùng Llama model

chat_router = APIRouter()
llm = LlamaChatModel()  # ✅ Khởi tạo model khi module được load

def generate_topic_from_message(text: str) -> str:
    if not text:
        return "Cuộc trò chuyện mới"
    topic = text[:50].strip()
    topic = re.sub(r'\s+', ' ', topic)
    if len(text) > 50:
        topic += "..."
    return topic

@chat_router.post("/api/chat")
async def chat(msg: ChatMessage, request: Request):
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}

    conversation_id = msg.conversation_id
    current_time = datetime.utcnow()
    
    # Create new conversation if needed
    if not conversation_id:
        topic = generate_topic_from_message(msg.text)
        new_conversation = {
            "user_email": user["email"],
            "topic": topic,
            "created_at": current_time,
            "last_message_at": current_time,
            "message_count": 0
        }
        conversation_result = await conversation_collection.insert_one(new_conversation)
        conversation_id = str(conversation_result.inserted_id)

    # Save user message
    await chat_collection.insert_one({
        "user_email": user["email"],
        "conversation_id": conversation_id,
        "role": "user",
        "text": msg.text,
        "images": msg.images,
        "timestamp": current_time
    })

    # ✅ Sinh trả lời từ LLM
    bot_reply = llm.generate(msg.text)

    # Save bot reply
    await chat_collection.insert_one({
        "user_email": user["email"],
        "conversation_id": conversation_id,
        "role": "bot",
        "text": bot_reply,
        "images": [],
        "timestamp": current_time
    })

    # Update conversation stats
    await conversation_collection.update_one(
        {"_id": ObjectId(conversation_id)},
        {
            "$set": {"last_message_at": current_time},
            "$inc": {"message_count": 2}
        }
    )

    return {"reply": bot_reply, "conversation_id": conversation_id}
