# ivr_conversational.py
"""
Unified IVR backend with conversational AI flows.
- FastAPI app providing:
  * /ivr/start        -> start a call session
  * /ivr/dtmf         -> legacy DTMF handling
  * /ivr/sim_speech   -> simulate STT arriving (text)
  * /ivr/stt_callback -> real STT webhook (from ACS/Twilio) - call into conversation handler
  * /ivr/end          -> end call
  * /ivr/history      -> call logs
- Replace provider placeholders (STT/TTS/LLM) with real implementations.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import asyncio
import re
import os

app = FastAPI(title="IVR Conversational Backend")

# -------------------------
# Configuration toggles
# -------------------------
LLM_ENABLED = False               # set True to enable LLM-based NLU (must implement llm_nlu_parse)
LLM_API_KEY = os.getenv("LLM_API_KEY", None)

# -------------------------
# In-memory state (simple)
# -------------------------
active_calls: Dict[str, Dict[str, Any]] = {}
call_history: list = []

# -------------------------
# Legacy IVR Menu (you can expand)
# -------------------------
MENU_STRUCTURE = {
    "main": {
        "prompt": "Welcome to Air India. Press 1 for Booking. Press 2 for Flight Status. Press 3 for Agent.",
        "options": {
            "1": {"action": "goto_menu", "target": "booking", "message": "Going to Booking."},
            "2": {"action": "goto_menu", "target": "flight_status", "message": "Going to Flight Status."},
            "3": {"action": "transfer_agent", "message": "Transferring to agent."}
        }
    },
    "booking": {
        "prompt": "Booking menu: Press 1 Domestic. Press 2 International. Press 0 Back.",
        "options": {
            "1": {"action": "start_booking_domestic", "message": "Domestic booking selected."},
            "2": {"action": "start_booking_international", "message": "International booking selected."},
            "0": {"action": "goto_menu", "target": "main", "message": "Returning to main menu."}
        }
    },
    "flight_status": {
        "prompt": "Please speak or enter 6-digit PNR. Press 0 Back.",
        "options": {
            "0": {"action": "goto_menu", "target": "main", "message": "Returning to main menu."}
        }
    }
}

# -------------------------
# Request models
# -------------------------
class StartCall(BaseModel):
    caller: str

class DTMFInput(BaseModel):
    call_id: str
    digit: str

class SimSpeech(BaseModel):
    call_id: str
    transcript: str

class EndCallModel(BaseModel):
    call_id: str

# -------------------------
# Util: small helpers
# -------------------------
def make_call_session(caller: str):
    call_id = str(uuid.uuid4())
    active_calls[call_id] = {
        "call_id": call_id,
        "caller": caller,
        "created_at": datetime.utcnow().isoformat(),
        "current_menu": "main",
        "menu_path": ["main"],
        "conv_mode": True,              # conversational mode enabled by default
        "nlp_context": {"history": [], "slots": {}},
        "pnr_buffer": "",               # accumulate digits if in PNR input
        "status": "active"
    }
    return call_id

def cleanup_call(call_id: str):
    c = active_calls.pop(call_id, None)
    if c:
        c["ended_at"] = datetime.utcnow().isoformat()
        call_history.append(c)
    return c

# -------------------------
# NLU: rule-based + optional LLM hook
# -------------------------
def rule_based_nlu(text: str) -> Dict[str, Any]:
    t = text.lower().strip()
    # simple patterns
    if any(x in t for x in ["book", "booking", "reserve", "ticket"]):
        return {"intent": "booking_enquiry", "confidence": 0.9, "entities": {}}
    if any(x in t for x in ["pnr", "status", "flight status", "where is my flight", "flight"]):
        # extract digits that look like a PNR (6 digits) OR token-like alphanumeric (fallback)
        digits = re.findall(r"\d+", t)
        pnr = None
        for d in digits:
            if len(d) == 6:
                pnr = d
                break
        return {"intent": "flight_status", "confidence": 0.85, "entities": {"pnr": pnr}}
    if any(x in t for x in ["agent", "human", "representative", "operator"]):
        return {"intent": "agent_transfer", "confidence": 0.95, "entities": {}}
    if any(x in t for x in ["hi","hello","hey","good morning","good evening"]):
        return {"intent": "greeting", "confidence": 0.8, "entities": {}}
    if any(x in t for x in ["bye", "goodbye", "thanks", "thank you"]):
        return {"intent": "end_call", "confidence": 0.95, "entities": {}}
    # fallback
    return {"intent": "unknown", "confidence": 0.4, "entities": {}}

async def llm_nlu_parse(text: str) -> Dict[str, Any]:
    """
    Replace this with your LLM/AI call (OpenAI/Azure/Open-Source). Should return a dict with:
      {"intent": "...", "confidence": 0.9, "entities": {...}}
    For now, calls rule_based_nlu as fallback.
    """
    # Example placeholder: call external LLM here using LLM_API_KEY
    # e.g., openai.ChatCompletion.create(...) or Azure OpenAI
    # keep this async for network I/O compatibility
    await asyncio.sleep(0)  # no-op to keep function async
    return rule_based_nlu(text)

async def nlu_parse(text: str) -> Dict[str, Any]:
    if LLM_ENABLED and LLM_API_KEY:
        return await llm_nlu_parse(text)
    return rule_based_nlu(text)

# -------------------------
# Dialog Manager & Intent mapping
# -------------------------
INTENT_ACTION_MAP = {
    "booking_enquiry": {"action": "goto_menu", "target_menu": "booking", "response": "Sure — I can help with your booking. Domestic or international?"},
    "flight_status": {"action": "flight_status_check", "target_menu": "flight_status", "response": "Please tell me your 6-digit PNR."},
    "agent_transfer": {"action": "transfer_agent", "response": "I'll transfer you to an agent now."},
    "greeting": {"action": "none", "response": "Hello! How can I help you today?"},
    "end_call": {"action": "end_call", "response": "Thanks for calling. Goodbye!"},
    "unknown": {"action": "reprompt", "response": "Sorry, I didn't understand. Could you please repeat?"}
}

async def map_intent_to_action(nlu_result: Dict[str, Any], call_session: Dict[str, Any]) -> Dict[str, Any]:
    intent = nlu_result.get("intent")
    mapping = INTENT_ACTION_MAP.get(intent, INTENT_ACTION_MAP["unknown"])

    # Flight status with PNR entity -> perform lookup inline
    if intent == "flight_status":
        pnr = nlu_result.get("entities", {}).get("pnr")
        if pnr:
            # call your real PNR lookup here. For demo, mock.
            flight_info = {"pnr": pnr, "flight": "AI101", "status": "Confirmed", "route": "Mumbai-Delhi"}
            return {"action": "speak_and_hangup", "response": f"PNR {pnr} is confirmed. Flight {flight_info['flight']} from {flight_info['route']} is {flight_info['status']}."}
        # else require PNR slot
        return {"action": "ask_for_pnr", "response": mapping["response"]}

    # Booking: map to booking menu (reuse legacy menus)
    if intent == "booking_enquiry":
        return {"action": "goto_menu", "target_menu": "booking", "response": mapping["response"]}

    if intent == "agent_transfer":
        return {"action": "transfer_agent", "response": mapping["response"]}

    if intent == "end_call":
        return {"action": "speak_and_hangup", "response": mapping["response"]}

    if intent == "greeting":
        return {"action": "speak", "response": mapping["response"]}

    return {"action": "reprompt", "response": mapping["response"]}

# -------------------------
# STT/TTS Provider placeholders (replace with real provider)
# -------------------------
async def play_tts_to_call(call_id: str, text: str) -> None:
    """
    Send TTS audio back to a live call connection.
    Replace with provider-specific code:
      - Azure Communication Services: call_connection.play_media(...)
      - Twilio: <Play> with TTS URL or stream audio
      - Google: stream audio back via WebRTC or telephony bridge
    This placeholder simply logs and simulates a small delay.
    """
    print(f"[TTS] to {call_id}: {text}")
    # simulate playback duration proportional to text length
    await asyncio.sleep(min(3.0, max(0.5, len(text) / 80.0)))

async def hangup_call(call_id: str):
    print(f"[CALL] Hanging up {call_id}")
    # place provider hang-up code here
    cleanup_call(call_id)

# -------------------------
# Conversational handler (core)
# -------------------------
async def handle_transcribed_text(call_id: str, transcript: str) -> Dict[str, Any]:
    """
    Core conversational flow:
      - Run NLU (rule or LLM)
      - Map intent to action
      - Call legacy menu handlers or perform action (PNR lookup / transfer)
      - Play TTS back (placeholder)
    Returns a dictionary of outcome for testing.
    """
    session = active_calls.get(call_id)
    if not session:
        raise ValueError("Unknown call_id")

    # store transcript
    session["nlp_context"]["history"].append({"from": "user", "text": transcript, "ts": datetime.utcnow().isoformat()})

    # Run NLU
    nlu_result = await nlu_parse(transcript)
    session["nlp_context"]["last_nlu"] = nlu_result

    # Map to action
    action = await map_intent_to_action(nlu_result, session)

    # Execute action
    act = action.get("action")
    resp_text = action.get("response", "")

    # ACTIONS:
    if act == "goto_menu":
        target = action.get("target_menu")
        session["current_menu"] = target
        session["menu_path"].append(target)
        menu_prompt = MENU_STRUCTURE.get(target, {}).get("prompt", resp_text)
        await play_tts_to_call(call_id, menu_prompt)
        return {"status": "ok", "action": "goto_menu", "menu": target, "spoken": menu_prompt}

    if act == "ask_for_pnr":
        # change session so subsequent speech/dtmf is captured as PNR digits
        session["current_menu"] = "flight_status"
        await play_tts_to_call(call_id, resp_text)
        return {"status": "ok", "action": "ask_for_pnr", "spoken": resp_text}

    if act == "speak":
        await play_tts_to_call(call_id, resp_text)
        return {"status": "ok", "action": "speak", "spoken": resp_text}

    if act == "speak_and_hangup":
        await play_tts_to_call(call_id, resp_text)
        await asyncio.sleep(0.3)
        await hangup_call(call_id)
        return {"status": "ok", "action": "hangup", "spoken": resp_text}

    if act == "transfer_agent":
        await play_tts_to_call(call_id, resp_text)
        # insert provider-specific transfer logic here (bridge to human queue)
        session["status"] = "transferring"
        # optionally hangup local leg if PBX handles transfer
        return {"status": "ok", "action": "transfer", "spoken": resp_text}

    if act == "reprompt" or act == "unknown":
        await play_tts_to_call(call_id, resp_text)
        return {"status": "ok", "action": "reprompt", "spoken": resp_text}

    # default fallback
    await play_tts_to_call(call_id, "Sorry, I couldn't handle that. Transferring to agent.")
    return {"status": "fail", "action": "transfer", "spoken": "transferring"}
# -------------------------
# REAL SPEECH-TO-TEXT (AZURE SPEECH SDK)
# -------------------------

import azure.cognitiveservices.speech as speechsdk

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY,
    region=AZURE_SPEECH_REGION
)
speech_config.speech_recognition_language = "en-IN"

async def transcribe_speech() -> str:
    """
    Real-time STT using Azure Speech SDK.
    Works for mic input or telephony streamed audio.
    """
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("🎤 Listening... speak now...")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"✅ Recognized: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return ""
    else:
        return ""
@app.post("/ivr/live_stt")
async def live_stt(request: Request):
    """
    Real-time mic STT injection. Test endpoint.
    It listens from mic, transcribes, injects into IVR pipeline.
    """
    body = await request.json()
    call_id = body.get("call_id")

    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")

    transcript = await transcribe_speech()

    if not transcript:
        return {"status": "no_speech_detected"}

    result = await handle_transcribed_text(call_id, transcript)
    return {
        "stt_text": transcript,
        "ivr_response": result
    }


# -------------------------
# REST endpoints
# -------------------------
@app.post("/ivr/start")
def start_call(req: StartCall):
    call_id = make_call_session(req.caller)
    prompt = MENU_STRUCTURE["main"]["prompt"]
    # In real integration: you might immediately play TTS to the call via provider
    return {"call_id": call_id, "initial_prompt": prompt}

@app.post("/ivr/dtmf")
def dtmf_endpoint(inp: DTMFInput):
    """
    Handle legacy DTMF input. Uses MENU_STRUCTURE mapping and also integrates
    with conversational mode (e.g., entering digits while asking for PNR).
    """
    call = active_calls.get(inp.call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    menu = call["current_menu"]
    digit = inp.digit

    # If inside flight_status expecting digits, accumulate
    if menu == "flight_status" and digit.isdigit():
        call["pnr_buffer"] += digit
        if len(call["pnr_buffer"]) == 6:
            pnr = call["pnr_buffer"]
            call["pnr_buffer"] = ""
            # mock lookup
            response = f"PNR {pnr} is confirmed. Flight AI101 is on time."
            # play and hangup
            asyncio.create_task(play_tts_to_call(inp.call_id, response))
            asyncio.create_task(hangup_call(inp.call_id))
            return {"action": "pnr_lookup", "message": response}

        # else ask for more digits
        remaining = 6 - len(call["pnr_buffer"])
        return {"message": f"Received digit. {remaining} digits to go."}

    # Standard menu option handling
    opt = MENU_STRUCTURE.get(menu, {}).get("options", {}).get(digit)
    if not opt:
        return {"message": "Invalid choice, please try again."}

    if opt["action"] == "goto_menu":
        call["current_menu"] = opt["target"]
        call["menu_path"].append(opt["target"])
        prompt = MENU_STRUCTURE[opt["target"]]["prompt"]
        # in production, play TTS to the call; here just return prompt
        asyncio.create_task(play_tts_to_call(inp.call_id, prompt))
        return {"message": prompt}

    if opt["action"] == "transfer_agent":
        asyncio.create_task(play_tts_to_call(inp.call_id, opt.get("message", "Transferring")))
        # provider-specific transfer logic goes here
        call["status"] = "transferring"
        return {"message": "transfer initiated"}

    if opt["action"] == "end_call":
        asyncio.create_task(play_tts_to_call(inp.call_id, opt.get("message", "Goodbye")))
        asyncio.create_task(hangup_call(inp.call_id))
        return {"message": "call ended"}

    return {"message": "Unhandled DTMF action"}

@app.post("/ivr/sim_speech")
async def simulate_speech(inp: SimSpeech):
    """
    Simulator endpoint: use to emulate STT arriving from a real call.
    This uses the same conversational pipeline as /ivr/stt_callback.
    """
    if inp.call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not active")
    res = await handle_transcribed_text(inp.call_id, inp.transcript)
    return res

@app.post("/ivr/stt_callback")
async def stt_callback(request: Request):
    """
    Generic STT callback receiver for real STT providers (ACS/Twilio).
    The exact payload depends on provider:
    - Azure Communication Services: RecognizeSpeechCompleted events contain recognized text.
    - Twilio: use <Stream> or speech recognition webhook.
    Map provider payload to {"call_id":..., "transcript": "..."}
    """
    payload = await request.json()
    # Provider-specific mapping required. Here, we attempt some common shapes:
    call_id = None
    transcript = None

    # Example Azure event: { "type": "...", "recognizeResult": {"text": "..."}, "callConnectionId": "<id>"}
    if isinstance(payload, dict):
        if "callConnectionId" in payload and "recognizeResult" in payload:
            call_id = payload.get("callConnectionId")
            transcript = payload["recognizeResult"].get("text")
        # Twilio might post: {"CallSid": "...", "SpeechResult":"..."}
        if not transcript and "CallSid" in payload and "SpeechResult" in payload:
            call_id = payload.get("CallSid")
            transcript = payload.get("SpeechResult")
        # Generic: try keys
        if not transcript:
            transcript = payload.get("transcript") or payload.get("text") or payload.get("recognizedText")

    if not call_id or not transcript:
        # We couldn't parse provider payload; return 400 with diagnostics
        raise HTTPException(status_code=400, detail=f"Could not parse STT payload. Received keys: {list(payload.keys())}")

    # Ensure call exists
    if call_id not in active_calls:
        # sometimes call ids differ between provider and our internal uuid.
        # If you use provider call ids, store mapping when starting call.
        raise HTTPException(status_code=404, detail="Call session not found")

    # handle in background so webhook returns quickly
    asyncio.create_task(handle_transcribed_text(call_id, transcript))
    return {"status": "accepted"}

@app.post("/ivr/end")
def end_call(req: EndCallModel):
    cleanup_call(req.call_id)
    return {"status": "call ended"}

@app.get("/ivr/history")
def get_history():
    return {"history": call_history, "active_count": len(active_calls)}

# -------------------------
# Example run/test instructions (prints)
# -------------------------
if __name__ == "__main__":
    print("This module is a FastAPI app. Run with:")
    print("  uvicorn ivr_conversational:app --reload")
    print("Endpoints:")
    print("  POST /ivr/start   -> {caller: '+91...'}")
    print("  POST /ivr/dtmf    -> {call_id, digit}")
    print("  POST /ivr/sim_speech -> {call_id, transcript}")
    print("  POST /ivr/stt_callback -> provider webhook payload (map required)")
    print("  POST /ivr/end     -> {call_id}")

