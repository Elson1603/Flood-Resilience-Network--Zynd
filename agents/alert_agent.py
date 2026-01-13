from agents.base_agent import BaseAgent
from typing import Dict, Any, List
from fastapi import WebSocket

class AlertAgent(BaseAgent):
    def __init__(self, agent_id: str):
        capabilities = ["emergency_alert", "notification", "broadcast"]
        super().__init__(agent_id, "AlertAgent", capabilities)
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime
        self.status = "alerting"
        self.last_action = datetime.now()
        self.action_count += 1
        
        location_name = data.get('location', {}).get('name', 'Unknown')
        risk_level = data.get('risk_level', 'UNKNOWN')
        probability = data.get('flood_probability', 0)
        
        print(f"\n🚨 ALERT AGENT")
        print(f"   {risk_level} risk in {location_name} ({probability*100:.1f}%)")
        
        alert_message = {
            "type": "FLOOD_ALERT",
            "risk_level": risk_level,
            "location": data.get('location', {}),
            "probability": probability,
            "message": f"{risk_level} RISK: Flood probability {probability*100:.1f}% in {location_name}",
            "timestamp": data.get('timestamp')
        }
        
        await self.broadcast(alert_message)
        
        print(f"   ✓ Alert sent to {len(self.active_connections)} clients")
        
        self.status = "ready"
        return {"alerts_sent": len(self.active_connections)}
    
    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.active_connections.remove(conn)
