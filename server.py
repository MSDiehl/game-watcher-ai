import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Create FastAPI app
app = FastAPI()

# Data model for incoming events
class Event(BaseModel):
    type: str
    window: str
    details: dict
    log: str | None = None
    timestamp: float

@app.post("/event")
async def handle_event(event: Event):
    """
    Receives game/input events, builds a prompt with a snarky, rude persona,
    queries OpenAI, and returns the AI's response.
    """
    # System message guiding the AI persona
    system_msg = {
        "role": "system",
        "content": (
            "You are an irritated and rude game commentator. "
            "Provide short, biting commentary that conveys annoyance. "
            "You often use mild profanity like 'damn', 'hell', 'fuck', or 'shit' to emphasize your irritation, "
        )
    }

    # Construct the user message from event data
    lines = [
        f"Event Type: {event.type}",
        f"Game Window: {event.window}",
        f"Details: {event.details}",
    ]
    if event.log:
        lines.append(f"Log: {event.log}")
    lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}")
    user_msg = {"role": "user", "content": "\n".join(lines)}

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[system_msg, user_msg],
            max_tokens=100,
            temperature=0.8,
        )
        ai_text = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_text = f"[Error calling OpenAI API] {e}"

    return {"response": ai_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
