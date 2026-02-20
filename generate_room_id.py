from livekit import api
import os
from dotenv import load_dotenv

load_dotenv()

def get_join_token(room_name, participant_identity):
    token = api.AccessToken(
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET")
    )
    token.with_identity(participant_identity)
    token.with_name(participant_identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
    ))
    
    return token.to_jwt()


if __name__ == "__main__":
    room_name = "test-room"
    participant_identity = "test-participant"
    token = get_join_token(room_name, participant_identity)
    print("Join token:", token)
