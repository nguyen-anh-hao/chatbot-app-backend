from fastapi import APIRouter, Request
from database.mongodb import chat_collection

history_router = APIRouter()

@history_router.get("/api/history")
async def get_history(request: Request):
    user = request.session.get("user")
    if not user:
        return {"error": "Unauthorized"}

    cursor = chat_collection.find({"user_email": user["email"]})
    messages = await cursor.to_list(length=100)
    return [{"role": m["role"], "text": m.get("text"), "images": m.get("images")} for m in messages]
