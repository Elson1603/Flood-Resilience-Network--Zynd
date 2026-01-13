from agents.base_agent import BaseAgent
from typing import Dict, Any, List

class ResourceAgent(BaseAgent):
    def __init__(self, agent_id: str):
        capabilities = ["resource_allocation", "logistics", "coordination"]
        super().__init__(agent_id, "ResourceAgent", capabilities)
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime
        self.status = "mobilizing"
        self.last_action = datetime.now()
        self.action_count += 1
        
        location = data.get('location', {}).get('name', 'Unknown')
        risk_level = data.get('risk_level', 'LOW')
        
        print(f"\n💼 RESOURCE AGENT")
        print(f"   Mobilizing for {risk_level} risk in {location}")
        
        resources = self._get_resources(risk_level)
        
        print(f"   ✓ Allocated {len(resources)} resources")
        
        self.status = "ready"
        return {
            "resources_allocated": resources,
            "location": location,
            "risk_level": risk_level
        }
    
    def _get_resources(self, risk_level: str) -> List[Dict]:
        all_resources = [
            {"type": "rescue_team", "name": "NDRF Team Alpha", "personnel": 25},
            {"type": "rescue_team", "name": "Fire Brigade Unit", "personnel": 15},
            {"type": "medical", "name": "Mobile Hospital", "beds": 20},
            {"type": "equipment", "name": "Water Pumps", "capacity": "50000L/hr"},
            {"type": "shelter", "name": "Community Center", "capacity": 500},
            {"type": "shelter", "name": "School Building", "capacity": 300}
        ]
        
        if risk_level == "HIGH":
            return all_resources
        elif risk_level == "MEDIUM":
            return all_resources[:3]
        return []
