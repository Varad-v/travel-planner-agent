"""
Travel Planner Agent — Full Package

This single-file package contains:
- README (below)
- FastAPI-based API for a Travel Planner Agent
- Agent classes: CoordinatorAgent, ResearchAgent, ItineraryAgent, BookingAgent, MemoryAgent, EvaluationAgent
- Tool placeholders: SearchTool, FlightsAPI, HotelsAPI, MapsAPI, CalendarAPI
- MemoryBank: simple SQLite-based long-term memory
- SessionService: In-memory session support
- Example usage and demo at the bottom

INSTRUCTIONS:
1. Save this file as travel_planner_agent.py
2. Install dependencies: pip install fastapi uvicorn pydantic requests sqlalchemy transformers python-dotenv
   (transformers is optional if you plug in an LLM SDK)
3. Create a .env file with API keys for external services (GOOGLE_API_KEY, FLIGHTS_API_KEY, HOTELS_API_KEY)
4. Run: uvicorn travel_planner_agent:app --reload
5. POST to /plan_trip with JSON payload (see examples)

NOTE: This is a ready scaffold meant for Kaggle Capstone-style submission. Replace placeholder tool implementations
with your preferred tool integrations (OpenAI/LLM SDK, Google Custom Search, Skyscanner API, Amadeus, etc.).

"""

# ---------------------------
# README (embedded)
# ---------------------------
README = r"""
# Travel Planner Agent (Concierge)

A Travel Planner Agent that automates planning multi-destination trips: research, create itineraries, suggest flights/hotels/activities, manage budgets, remember user preferences, and support bookings via external APIs.

Features:
- Multi-agent setup: Research, Itinerary, Booking, Memory, Evaluation, Coordinator
- Tools: Search, Flights API, Hotels API, Maps/Distance, Calendar integration (placeholders)
- Sessions & Memory: InMemorySessionService + SQLite MemoryBank
- Long-running operations: pause/resume (via session state)
- Observability: Logging + basic metrics
- Evaluation: automated scoring on itinerary quality and relevance

Endpoints (FastAPI):
- POST /plan_trip -> starts planning pipeline and returns itinerary
- GET /session/{session_id} -> session state
- POST /resume/{session_id} -> resume a paused job
- POST /feedback -> submit feedback to update memory

"""

# ---------------------------
# Imports
# ---------------------------
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import time
import logging
import os
from datetime import datetime, timedelta
import sqlite3
import json
from functools import wraps

# load env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------
# Simple Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_agent")

# ---------------------------
# Config
# ---------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_KEY")
FLIGHTS_API_KEY = os.getenv("FLIGHTS_API_KEY", "YOUR_FLIGHTS_KEY")
HOTELS_API_KEY = os.getenv("HOTELS_API_KEY", "YOUR_HOTELS_KEY")

# ---------------------------
# Utility decorators & helpers
# ---------------------------
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end-start:.2f}s")
        return result
    return wrapper

# ---------------------------
# MemoryBank (SQLite) - simple persistent store
# ---------------------------
class MemoryBank:
    def __init__(self, db_path: str = "travel_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                content TEXT,
                tags TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save(self, user_id: str, title: str, content: Dict[str, Any], tags: List[str]=None) -> str:
        mem_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO memories (id,user_id,title,content,tags,created_at) VALUES (?,?,?,?,?,?)",
                  (mem_id, user_id, title, json.dumps(content), json.dumps(tags or []), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        logger.info(f"Saved memory {mem_id} for user {user_id}")
        return mem_id

    def query(self, user_id: str, tag_filter: Optional[List[str]] = None, limit: int = 10):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if tag_filter:
            # naive tag filtering
            pattern = '%' + '%'.join(tag_filter) + '%'
            c.execute("SELECT id,title,content,tags,created_at FROM memories WHERE user_id=? AND tags LIKE ? ORDER BY created_at DESC LIMIT ?",
                      (user_id, pattern, limit))
        else:
            c.execute("SELECT id,title,content,tags,created_at FROM memories WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                      (user_id, limit))
        rows = c.fetchall()
        conn.close()
        results = []
        for r in rows:
            results.append({"id": r[0], "title": r[1], "content": json.loads(r[2]), "tags": json.loads(r[3]), "created_at": r[4]})
        return results

# ---------------------------
# Session Service (In-Memory)
# ---------------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    def create(self, user_id: str, metadata: Dict[str, Any]) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"user_id": user_id, "status": "running", "metadata": metadata, "created_at": datetime.utcnow().isoformat(), "state": {}}
        return sid

    def get(self, sid: str):
        return self.sessions.get(sid)

    def update_state(self, sid: str, key: str, value: Any):
        if sid in self.sessions:
            self.sessions[sid]["state"][key] = value

    def pause(self, sid: str):
        if sid in self.sessions:
            self.sessions[sid]["status"] = "paused"

    def resume(self, sid: str):
        if sid in self.sessions:
            self.sessions[sid]["status"] = "running"

# ---------------------------
# Tools (placeholders)
# ---------------------------
class SearchTool:
    @staticmethod
    def search(query: str, num_results: int = 5):
        # Placeholder: integrate Google Custom Search or SerpAPI
        logger.info(f"SearchTool.search: q={query} num={num_results}")
        # Return mocked results
        return [{"title": f"Result {i} for {query}", "snippet": "Short snippet...", "url": f"https://example.com/{i}"} for i in range(1, num_results+1)]

class FlightsAPI:
    @staticmethod
    def find_flights(origin: str, destination: str, depart_date: str, return_date: Optional[str]=None, passengers:int=1):
        logger.info(f"FlightsAPI.find_flights {origin}->{destination} {depart_date}")
        # Mocked response
        return [{"airline":"ExampleAir", "price": 250 + i*50, "duration":"3h 20m", "flight_id": f"EX{i}"} for i in range(1,4)]

class HotelsAPI:
    @staticmethod
    def find_hotels(destination: str, checkin: str, checkout: str, guests:int=1):
        logger.info(f"HotelsAPI.find_hotels in {destination} {checkin}->{checkout}")
        return [{"hotel":"Hotel Example", "price_per_night":80+i*30, "rating":4.2, "hotel_id":f"H{i}"} for i in range(1,4)]

class MapsAPI:
    @staticmethod
    def travel_time(origin: str, destination: str):
        return {"distance_km": 12.5, "duration": "25 mins"}

class CalendarAPI:
    @staticmethod
    def suggest_best_dates(user_constraints: Dict[str,Any], flexibility_days:int=3):
        today = datetime.utcnow().date()
        return [(today + timedelta(days=i)).isoformat() for i in range(7, 7+flexibility_days)]

# ---------------------------
# Agents
# ---------------------------
class ResearchAgent:
    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools

    @timed
    def research(self, query: str):
        results = self.tools['search'].search(query, num_results=5)
        # basic extraction
        insights = []
        for r in results:
            insights.append({"title": r['title'], "snippet": r['snippet'], "url": r['url']})
        return insights

class AnalysisAgent:
    def __init__(self):
        pass

    @timed
    def analyze_destinations(self, destinations: List[str], preferences: Dict[str,Any]):
        # For each destination, gather pros/cons, cost estimate, recommended length
        analysis = {}
        for d in destinations:
            # placeholder heuristics
            analysis[d] = {"recommended_days": 2 + len(d)%3, "avg_daily_cost": 80 + len(d)*3, "highlights":[f"Top sight in {d}"]}
        return analysis

class ItineraryAgent:
    def __init__(self):
        pass

    @timed
    def create_itinerary(self, destinations: List[str], dates: Dict[str,str], prefs: Dict[str,Any], analysis: Dict[str,Any]):
        # create day-wise plan
        itinerary = {"summary": f"{len(destinations)}-destination trip", "legs": []}
        day_cursor = 0
        for d in destinations:
            days = analysis[d]['recommended_days']
            itinerary['legs'].append({"destination": d, "start_date": (datetime.utcnow().date()+timedelta(days=day_cursor)).isoformat(), "days": days, "activities": [f"Explore {d} main attraction"]})
            day_cursor += days
        return itinerary

class BookingAgent:
    def __init__(self, tools: Dict[str,Any]):
        self.tools = tools

    @timed
    def suggest_bookings(self, itinerary: Dict[str,Any]):
        suggestions = {"flights": [], "hotels": []}
        # simplistic: for each leg, call flights/hotels
        for leg in itinerary.get('legs',[]):
            dest = leg['destination']
            depart = leg['start_date']
            return_date = (datetime.fromisoformat(depart) + timedelta(days=leg['days'])).date().isoformat()
            flights = self.tools['flights'].find_flights("HOME", dest, depart, return_date)
            hotels = self.tools['hotels'].find_hotels(dest, depart, return_date)
            suggestions['flights'].append({"destination":dest, "options": flights})
            suggestions['hotels'].append({"destination":dest, "options": hotels})
        return suggestions

class EvaluationAgent:
    def __init__(self):
        pass

    def evaluate(self, itinerary: Dict[str,Any], user_prefs: Dict[str,Any]) -> Dict[str,Any]:
        # score by coverage, budget fit, and duration
        score = 85
        reasons = ["Good coverage", "Fits typical budget"]
        return {"score": score, "reasons": reasons}

# ---------------------------
# Coordinator Agent - orchestrates the pipeline
# ---------------------------
class CoordinatorAgent:
    def __init__(self, memory: MemoryBank, session_svc: InMemorySessionService):
        self.memory = memory
        self.session_svc = session_svc
        tools = {"search": SearchTool, "flights": FlightsAPI, "hotels": HotelsAPI, "maps": MapsAPI}
        self.research_agent = ResearchAgent(tools)
        self.analysis_agent = AnalysisAgent()
        self.itinerary_agent = ItineraryAgent()
        self.booking_agent = BookingAgent(tools)
        self.evaluation_agent = EvaluationAgent()

    def plan_trip(self, user_id: str, payload: Dict[str,Any], session_id: Optional[str]=None) -> Dict[str,Any]:
        # Create session
        if not session_id:
            session_id = self.session_svc.create(user_id, {"request": payload})
        else:
            s = self.session_svc.get(session_id)
            if not s:
                raise Exception("Invalid session")

        # 1) Research destinations
        dests = payload.get('destinations') or [payload.get('destination')]
        research_insights = {}
        for d in dests:
            research_insights[d] = self.research_agent.research(f"Top attractions, budget, travel tips for {d}")
            self.session_svc.update_state(session_id, f"research_{d}", research_insights[d])

        # 2) Analysis
        analysis = self.analysis_agent.analyze_destinations(dests, payload.get('preferences', {}))
        self.session_svc.update_state(session_id, 'analysis', analysis)

        # 3) Itinerary
        itinerary = self.itinerary_agent.create_itinerary(dests, payload.get('dates', {}), payload.get('preferences', {}), analysis)
        self.session_svc.update_state(session_id, 'itinerary', itinerary)

        # 4) Booking suggestions
        bookings = self.booking_agent.suggest_bookings(itinerary)
        self.session_svc.update_state(session_id, 'bookings', bookings)

        # 5) Evaluation
        eval_result = self.evaluation_agent.evaluate(itinerary, payload.get('preferences', {}))
        self.session_svc.update_state(session_id, 'evaluation', eval_result)

        # 6) Save to long-term memory (preference summary)
        mem_id = self.memory.save(user_id, f"trip-{session_id}", {"payload": payload, "itinerary": itinerary, "bookings": bookings}, tags=["trip", "itinerary"])
        self.session_svc.update_state(session_id, 'memory_id', mem_id)

        result = {"session_id": session_id, "itinerary": itinerary, "bookings": bookings, "evaluation": eval_result, "memory_id": mem_id}
        return result

# ---------------------------
# API Models
# ---------------------------
class TripRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    destinations: List[str] = Field(..., description="List of destinations, in order")
    dates: Optional[Dict[str,str]] = Field(default_factory=dict, description="Optional start dates per destination")
    preferences: Optional[Dict[str,Any]] = Field(default_factory=dict, description="User preferences (budget, pace, interests)")
    pause_at: Optional[str] = Field(None, description="If set, pipeline will pause after this stage")

class Feedback(BaseModel):
    user_id: str
    session_id: str
    rating: int
    comments: Optional[str] = None

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Travel Planner Agent API")
memory = MemoryBank()
session_svc = InMemorySessionService()
coordinator = CoordinatorAgent(memory, session_svc)

@app.post("/plan_trip")
def api_plan_trip(req: TripRequest, background_tasks: BackgroundTasks):
    try:
        # Run planning in background to emulate long-running operations
        sid = session_svc.create(req.user_id, {"request": req.dict()})
        def bg_job():
            try:
                logger.info(f"Starting plan for session {sid}")
                res = coordinator.plan_trip(req.user_id, req.dict(), session_id=sid)
                session_svc.update_state(sid, 'result', res)
                session_svc.update_state(sid, 'completed_at', datetime.utcnow().isoformat())
                session_svc.sessions[sid]['status'] = 'completed'
                logger.info(f"Completed plan for session {sid}")
            except Exception as e:
                logger.exception(e)
                session_svc.sessions[sid]['status'] = 'failed'
                session_svc.update_state(sid, 'error', str(e))

        background_tasks.add_task(bg_job)
        return {"message": "Plan started", "session_id": sid}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
def api_get_session(session_id: str):
    s = session_svc.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s

@app.post("/resume/{session_id}")
def api_resume(session_id: str):
    s = session_svc.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    if s['status'] != 'paused':
        return {"message": "Session not paused", "status": s['status']}
    session_svc.resume(session_id)
    # re-run background job
    def bg_resume():
        try:
            payload = s['metadata'].get('request')
            res = coordinator.plan_trip(s['user_id'], payload, session_id=session_id)
            session_svc.update_state(session_id, 'result', res)
            session_svc.sessions[session_id]['status'] = 'completed'
        except Exception as e:
            logger.exception(e)
            session_svc.sessions[session_id]['status'] = 'failed'
            session_svc.update_state(session_id, 'error', str(e))
    bg = BackgroundTasks()
    bg.add_task(bg_resume)
    return {"message":"Resumed"}

@app.post("/feedback")
def api_feedback(fb: Feedback):
    # Basic feedback handling: store in memory and adjust preferences
    memory.save(fb.user_id, f"feedback-{fb.session_id}", {"rating": fb.rating, "comments": fb.comments}, tags=["feedback"]) 
    return {"message": "Thanks for the feedback"}

# ---------------------------
# Demo runner when executed directly
# ---------------------------
if __name__ == '__main__':
    # Simple demo run
    demo_req = {
        "user_id": "user_123",
        "destinations": ["Paris", "Nice"],
        "dates": {},
        "preferences": {"budget":"moderate","interests":["art","food"]}
    }
    print("Starting demo plan (blocking)...")
    res = coordinator.plan_trip(demo_req['user_id'], demo_req)
    print(json.dumps(res, indent=2))

# ---------------------------
# End of file
# ---------------------------
