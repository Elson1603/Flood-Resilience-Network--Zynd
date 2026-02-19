import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import asyncio
import os
from datetime import datetime

from agents.monitoring_agent import MonitoringAgent
from agents.prediction_agent import PredictionAgent
from agents.coordination_agent import CoordinationAgent
from agents.alert_agent import AlertAgent
from agents.resource_agent import ResourceAgent

# Initialize FastAPI
app = FastAPI(
    title="Flood Resilience Network - Zynd AI",
    description="AI-Powered Multi-Agent Flood Prediction & Emergency Coordination",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
print("\n" + "="*80)
print("🚀 INITIALIZING FLOOD RESILIENCE NETWORK v2.0")
print("="*80 + "\n")

monitoring_agent = MonitoringAgent("monitor-001")
prediction_agent = PredictionAgent("predictor-001", model_path="models/flood_model_best.pth")
alert_agent = AlertAgent("alert-001")
resource_agent = ResourceAgent("resource-001")

agents = {
    "monitoring": monitoring_agent,
    "prediction": prediction_agent,
    "alert": alert_agent,
    "resource": resource_agent
}

coordination_agent = CoordinationAgent("coordinator-001", agents)

print("\n" + "="*80)
print("✓ ALL AGENTS INITIALIZED — SYSTEM OPERATIONAL")
print("="*80 + "\n")

# ── In-memory stores ──
prediction_history: List[Dict] = []
platform_stats = {
    "total_predictions": 0,
    "high_risk_count": 0,
    "medium_risk_count": 0,
    "low_risk_count": 0,
    "start_time": datetime.now().isoformat(),
}

# Known flood-prone zones in India
FLOOD_ZONES = [
    {"name": "Mumbai", "lat": 19.076, "lon": 72.877, "risk": "HIGH", "reason": "Coastal low-lying city, heavy monsoon rainfall", "population": "20.7M"},
    {"name": "Chennai", "lat": 13.083, "lon": 80.270, "risk": "HIGH", "reason": "Cyclone-prone coast, poor drainage", "population": "10.9M"},
    {"name": "Kolkata", "lat": 22.572, "lon": 88.363, "risk": "HIGH", "reason": "Ganges delta, below sea level areas", "population": "14.8M"},
    {"name": "Patna", "lat": 25.612, "lon": 85.144, "risk": "HIGH", "reason": "On the banks of river Ganges", "population": "2.5M"},
    {"name": "Guwahati", "lat": 26.144, "lon": 91.736, "risk": "HIGH", "reason": "Brahmaputra river floods annually", "population": "1.1M"},
    {"name": "Kochi", "lat": 9.931, "lon": 76.267, "risk": "HIGH", "reason": "Kerala backwaters, extreme rainfall", "population": "2.1M"},
    {"name": "Varanasi", "lat": 25.321, "lon": 83.010, "risk": "MEDIUM", "reason": "Ganges river proximity", "population": "1.4M"},
    {"name": "Hyderabad", "lat": 17.385, "lon": 78.486, "risk": "MEDIUM", "reason": "Urban flooding, Musi river", "population": "10.0M"},
    {"name": "Ahmedabad", "lat": 23.022, "lon": 72.571, "risk": "MEDIUM", "reason": "Sabarmati river, monsoon flooding", "population": "8.0M"},
    {"name": "Srinagar", "lat": 34.083, "lon": 74.797, "risk": "MEDIUM", "reason": "Jhelum river, glacier melt floods", "population": "1.7M"},
    {"name": "Delhi", "lat": 28.613, "lon": 77.209, "risk": "MEDIUM", "reason": "Yamuna river floods during monsoon", "population": "32.0M"},
    {"name": "Bengaluru", "lat": 12.971, "lon": 77.594, "risk": "LOW", "reason": "Elevated plateau, moderate rainfall", "population": "12.3M"},
    {"name": "Jaipur", "lat": 26.912, "lon": 75.787, "risk": "LOW", "reason": "Semi-arid climate, occasional flash floods", "population": "4.0M"},
    {"name": "Pune", "lat": 18.520, "lon": 73.856, "risk": "LOW", "reason": "Elevated terrain, dam-controlled rivers", "population": "7.4M"},
]

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Core Endpoints ──

@app.get("/")
async def root():
    return {
        "project": "Flood Resilience Network",
        "version": "2.0.0",
        "status": "operational",
        "agents": list(agents.keys()),
        "total_predictions": platform_stats["total_predictions"],
    }

@app.get("/demo")
async def demo_page():
    """Serve interactive dashboard"""
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_flood(location: Dict[str, Any]):
    """Predict flood risk for a location — orchestrates all agents"""
    result = await coordination_agent.execute(location)
    
    # Update stats
    risk = result.get("risk_level", "LOW")
    platform_stats["total_predictions"] += 1
    if risk == "HIGH":
        platform_stats["high_risk_count"] += 1
    elif risk == "MEDIUM":
        platform_stats["medium_risk_count"] += 1
    else:
        platform_stats["low_risk_count"] += 1
    
    # Store in history
    history_entry = {
        "id": platform_stats["total_predictions"],
        "location": result.get("location", {}).get("name", "Unknown"),
        "probability": result.get("flood_probability", 0),
        "risk_level": risk,
        "timestamp": result.get("timestamp", datetime.now().isoformat()),
    }
    prediction_history.insert(0, history_entry)
    
    # Keep only last 50 entries
    if len(prediction_history) > 50:
        prediction_history.pop()
    
    return JSONResponse(content=result)

# ── New Endpoints ──

@app.get("/api/stats")
async def get_stats():
    """Platform statistics"""
    return JSONResponse(content={
        "total_predictions": platform_stats["total_predictions"],
        "high_risk_count": platform_stats["high_risk_count"],
        "medium_risk_count": platform_stats["medium_risk_count"],
        "low_risk_count": platform_stats["low_risk_count"],
        "active_agents": len(agents),
        "uptime_since": platform_stats["start_time"],
        "model_type": "MLP Neural Network",
        "features_count": 20,
    })

@app.get("/api/history")
async def get_history():
    """Prediction history"""
    return JSONResponse(content=prediction_history)

@app.get("/api/flood-zones")
async def get_flood_zones():
    """Known flood-prone zones"""
    return JSONResponse(content=FLOOD_ZONES)

@app.get("/api/weather/{lat}/{lon}")
async def get_weather(lat: float, lon: float):
    """Get simulated weather data for coordinates"""
    weather = monitoring_agent.get_simulated_weather(lat, lon, rainfall_mm=50)
    return JSONResponse(content=weather)

@app.get("/api/safety-tips/{risk_level}")
async def get_safety_tips(risk_level: str):
    """Get safety tips for a risk level"""
    tips = resource_agent._get_safety_tips(risk_level.upper())
    contacts = resource_agent._get_emergency_contacts()
    evacuation = resource_agent._get_evacuation_info(risk_level.upper(), "General")
    return JSONResponse(content={
        "risk_level": risk_level.upper(),
        "safety_tips": tips,
        "emergency_contacts": contacts,
        "evacuation": evacuation,
    })

@app.get("/agent-status")
async def get_agent_status():
    """Get status of all agents"""
    statuses = {}
    for name, agent in agents.items():
        statuses[name] = await agent.report_status()
    return JSONResponse(content=statuses)

@app.websocket("/ws/alerts/{client_id}")
async def websocket_alerts(websocket: WebSocket, client_id: str):
    """WebSocket for real-time alerts"""
    await alert_agent.connect(websocket)
    print(f"✓ Client {client_id} connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "status": "connected",
                "client_id": client_id,
                "message": "Monitoring for flood alerts..."
            })
    except WebSocketDisconnect:
        alert_agent.disconnect(websocket)
        print(f"⚠ Client {client_id} disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
