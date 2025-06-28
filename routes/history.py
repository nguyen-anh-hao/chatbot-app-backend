from fastapi import APIRouter, Request, Query
from database.mongodb import chat_collection, conversation_collection
from bson import ObjectId
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class ConversationCreate(BaseModel):
    topic: str

history_router = APIRouter()

@history_router.get("/api/conversations")
async def get_conversations(request: Request):
    """Get list of all conversations for the user"""
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}

    cursor = conversation_collection.find(
        {"user_email": user["email"]}
    ).sort("last_message_at", -1)
    
    conversations = await cursor.to_list(length=100)
    
    return [
        {
            "id": str(conv["_id"]),
            "topic": conv["topic"],
            "created_at": conv["created_at"],
            "last_message_at": conv["last_message_at"],
            "message_count": conv["message_count"]
        }
        for conv in conversations
    ]

@history_router.get("/api/history")
async def get_history(request: Request, conversation_id: Optional[str] = Query(None)):
    """Get chat history for a specific conversation or all messages"""
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}

    query = {"user_email": user["email"]}
    if conversation_id:
        query["conversation_id"] = conversation_id

    cursor = chat_collection.find(query).sort("timestamp", 1)
    messages = await cursor.to_list(length=1000)
    
    return [
        {
            "role": m["role"],
            "text": m.get("text"),
            "images": m.get("images"),
            "conversation_id": m.get("conversation_id"),
            "timestamp": m.get("timestamp")
        }
        for m in messages
    ]

@history_router.post("/api/conversations")
async def create_conversation(conversation_data: ConversationCreate, request: Request):
    """Create a new conversation"""
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}

    current_time = datetime.utcnow()
    
    new_conversation = {
        "user_email": user["email"],
        "topic": conversation_data.topic,
        "created_at": current_time,
        "last_message_at": current_time,
        "message_count": 0
    }
    
    result = await conversation_collection.insert_one(new_conversation)
    
    return {
        "id": str(result.inserted_id),
        "topic": conversation_data.topic,
        "created_at": current_time,
        "last_message_at": current_time,
        "message_count": 0
    }

@history_router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request):
    """Delete a conversation and all its messages"""
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}
    
    # Verify conversation belongs to user
    conversation = await conversation_collection.find_one({
        "_id": ObjectId(conversation_id),
        "user_email": user["email"]
    })
    
    if not conversation:
        return {"error": "Conversation not found"}
    
    # Delete all messages in the conversation
    await chat_collection.delete_many({
        "conversation_id": conversation_id,
        "user_email": user["email"]
    })
    
    # Delete the conversation
    await conversation_collection.delete_one({
        "_id": ObjectId(conversation_id)
    })
    
    return {"message": "Conversation deleted successfully"}

@history_router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, request: Request):
    """Get messages for a specific conversation"""
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}

    # Verify conversation belongs to user
    conversation = await conversation_collection.find_one({
        "_id": ObjectId(conversation_id),
        "user_email": user["email"]
    })
    
    if not conversation:
        return {"error": "Conversation not found"}

    cursor = chat_collection.find({
        "conversation_id": conversation_id,
        "user_email": user["email"]
    }).sort("timestamp", 1)
    
    messages = await cursor.to_list(length=1000)
    
    return [
        {
            "role": m["role"],
            "text": m.get("text"),
            "images": m.get("images"),
            "conversation_id": m.get("conversation_id"),
            "timestamp": m.get("timestamp")
        }
        for m in messages
    ]
