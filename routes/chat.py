from fastapi import APIRouter, Request
from models.chat import ChatMessage
from database.mongodb import chat_collection, conversation_collection
from datetime import datetime
from bson import ObjectId
import re
from rag_production import RAGRecipeModel
# from llm.models import ModelHandler

chat_router = APIRouter()
# model_handler = ModelHandler()

def generate_topic_from_message(text: str) -> str:
    """Generate a simple topic from the first message"""
    if not text:
        return "Cuộc trò chuyện mới"
    
    # Take first 50 characters and clean it up
    topic = text[:50].strip()
    # Remove newlines and multiple spaces
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
    
    # If no conversation_id provided, create a new conversation
    if not conversation_id:
        topic = generate_topic_from_message(msg.text)
        
        # Create new conversation
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

    # model = model_handler.get_model()
    # results = model.search(msg.text, top_k=3)
    # bot_reply = f"\nTop 3 results for '{msg.text}':\n"
    # for result in results:
    #     bot_reply += f"{result['rank']}. {result['title']} (Score: {result['similarity_score']:.3f})\n"

    # bot_reply = "Đây là câu trả lời từ bot. Vui lòng tích hợp mô hình LLM của bạn để trả lời câu hỏi."
    model = RAGRecipeModel()
    model.load_model()
    results = model.search(msg.text, top_k=3)
    bot_reply = f"Top 3 results for '{msg.text}':\n"
    for result in results:
        bot_reply += f"{result['rank']}. {result['title']} (Score: {result['similarity_score']:.3f})\n"
    
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
            "$inc": {"message_count": 2}  # user message + bot reply
        }
    )

    return {"reply": bot_reply, "conversation_id": conversation_id}
