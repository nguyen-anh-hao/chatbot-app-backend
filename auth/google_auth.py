from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_RESPONSE
import os
import logging

logger = logging.getLogger(__name__)

google_router = APIRouter()
config = Config(environ=os.environ)

oauth = OAuth(config)
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

@google_router.get("/login")
async def login(request: Request):
    redirect_uri = str(request.url_for('auth'))
    print("🔁 Redirect URI used:", redirect_uri)
    return await oauth.google.authorize_redirect(request, redirect_uri)

@google_router.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        logger.info(f"Token received: {token.keys()}")
        
        # Try to parse id_token first, fallback to userinfo endpoint
        try:
            if 'id_token' in token:
                user = await oauth.google.parse_id_token(request, token)
            else:
                # Fallback: fetch user info using access token
                user_response = await oauth.google.get('https://www.googleapis.com/oauth2/v2/userinfo', token=token)
                user = user_response.json()
        except Exception as e:
            logger.error(f"Error parsing user info: {e}")
            # Fallback: fetch user info using access token
            user_response = await oauth.google.get('https://www.googleapis.com/oauth2/v2/userinfo', token=token)
            user = user_response.json()
        
        request.session['user'] = dict(user)
        return RedirectResponse(url=REDIRECT_RESPONSE)
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return RedirectResponse(url="/login?error=auth_failed")

@google_router.get("/me")
async def me(request: Request):
    return {"user": request.session.get("user")}

@google_router.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url=REDIRECT_RESPONSE)