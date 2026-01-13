from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import asyncio
import os

from agents.monitoring_agent import MonitoringAgent
from agents.prediction_agent import PredictionAgent
from agents.coordination_agent import CoordinationAgent
from agents.alert_agent import AlertAgent
from agents.resource_agent import ResourceAgent

# Initialize FastAPI
app = FastAPI(
    title="Flood Resilience Network - Zynd AI",
    description="AI-Powered Multi-Agent Flood Prediction & Emergency Coordination",
    version="1.0.0"
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
print("🚀 INITIALIZING FLOOD RESILIENCE NETWORK")
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
print("✓ ALL AGENTS INITIALIZED")
print("="*80 + "\n")

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {
        "project": "Flood Resilience Network",
        "status": "operational",
        "version": "1.0.0",
        "agents": list(agents.keys()),
        "endpoints": {
            "demo": "/demo",
            "predict": "/predict",
            "agent_status": "/agent-status",
            "websocket": "/ws/alerts/{client_id}"
        }
    }

@app.get("/demo")
async def demo_page():
    """Serve interactive dashboard"""
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_flood(location: Dict[str, Any]):
    """
    Predict flood risk for location
    
    Example:
    {
        "location": {
            "name": "Mumbai",
            "lat": 19.0760,
            "lon": 72.8777,
            "elevation_m": 14,
            "rainfall_mm": 600,
            "river_proximity": 1,
            "slope_deg": 0.5
        }
    }
    """
    result = await coordination_agent.execute(location)
    return JSONResponse(content=result)

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
