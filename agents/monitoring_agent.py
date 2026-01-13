from agents.base_agent import BaseAgent
from typing import Dict, Any
import datetime

class MonitoringAgent(BaseAgent):
    def __init__(self, agent_id: str):
        capabilities = ["weather_monitoring", "data_collection", "real_time_feeds"]
        super().__init__(agent_id, "MonitoringAgent", capabilities)
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.status = "monitoring"
        self.last_action = datetime.datetime.now()
        self.action_count += 1
        
        location = data.get("location", {})
        
        print(f"\n🔍 MONITORING AGENT")
        print(f"   Location: {location.get('name', 'Unknown')}")
        
        # Use form input data directly
        processed_data = {
            "elevation_m": location.get('elevation_m', 0),
            "rainfall_mm": location.get('rainfall_mm', 0),
            "river_proximity": location.get('river_proximity', 1),
            "slope_deg": location.get('slope_deg', 0),
            "location": location,
            "timestamp": datetime.datetime.now().isoformat(),
            "data_source": "Form Input"
        }
        
        print(f"   ✓ Data collected: {processed_data['elevation_m']}m, {processed_data['rainfall_mm']}mm")
        
        self.status = "idle"
        return processed_data
