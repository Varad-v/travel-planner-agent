🌍 IntelliTrip — Multi-Agent Travel Planner

This project is a multi-agent travel planning system built using FastAPI, sessions, long-term memory, and parallel reasoning agents.
It generates:

✔ Travel itineraries

✔ Flight suggestions

✔ Hotel options

✔ Research insights

✔ Evaluation score

✔ Stored memory ID


🚀 Features
🧠 Multi-Agent Architecture

The system includes:

Research Agent → Collects travel insights per destination

Analysis Agent → Evaluates cost, highlights & recommended days

Itinerary Agent → Creates day-by-day travel plan

Booking Agent → Mock flight & hotel suggestions

Evaluation Agent → Scores the final plan

Memory Agent → Stores trip memory in SQLite

🧰 Tools Used

FastAPI

Uvicorn

Custom mock search tool

JSON-based session state

Background task execution

SQLite memory storage

📁 Project Structure
├── travel_planner_agent.py       # Main backend (FastAPI + multi-agent logic)
├── travel_plan_ui_fixed.html     # Offline Web UI to view travel results
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── travel_memory.db              # Auto-generated memory database

▶️ Running the Backend
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Start the FastAPI server
uvicorn travel_planner_agent:app --reload


The API will start at:

Swagger UI → http://127.0.0.1:8000/docs

Redoc → http://127.0.0.1:8000/redoc

🧪 Test the System
🔹 Step 1: Start a planning session

POST → /plan_trip

Example JSON:

{
  "user_id": "user_123",
  "destinations": ["Paris", "Nice"],
  "preferences": {
    "budget": "moderate",
    "interests": ["art", "food"]
  }
}


This returns a session_id.

🔹 Step 2: Fetch results

GET → /session/{session_id}

This gives you:

itinerary

bookings

evaluation

research

memory_id

🎨 Viewing the Results (Offline Web UI)

Open:

travel_plan_ui_fixed.html


Paste your JSON output from /session/{session_id}

Click Render Plan

Your complete itinerary, flights, hotels, evaluation, and research appear in UI

Click Download PDF to export

🏁 Kaggle Capstone Requirements Covered

✔ Multi-Agent System
✔ Tools (custom search, memory DB)
✔ Parallel/sequential agents
✔ Sessions & state tracking
✔ Long-term memory
✔ Background tasks
✔ Evaluation agent
✔ Deployment-ready API

💡 Future Enhancements

Integrate live flight/hotel API

Deploy API on Render / HuggingFace

Add chat-style frontend

Add “Auto Fetch Session” UI

Improve memory storage using vector DB

👨‍💻 Author

Varad Khatavkar
Artificial Intelligence & Data Science
IntelliTrip — Multi-Agent Travel Planner
