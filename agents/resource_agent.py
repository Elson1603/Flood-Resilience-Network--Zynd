from agents.base_agent import BaseAgent
from typing import Dict, Any, List

class ResourceAgent(BaseAgent):
    def __init__(self, agent_id: str):
        capabilities = ["resource_allocation", "logistics", "coordination", "safety_advisory"]
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
        safety_tips = self._get_safety_tips(risk_level)
        emergency_contacts = self._get_emergency_contacts()
        evacuation = self._get_evacuation_info(risk_level, location)
        
        print(f"   ✓ Allocated {len(resources)} resources, {len(safety_tips['dos'])} safety tips")
        
        self.status = "ready"
        return {
            "resources_allocated": resources,
            "safety_tips": safety_tips,
            "emergency_contacts": emergency_contacts,
            "evacuation": evacuation,
            "location": location,
            "risk_level": risk_level
        }
    
    def _get_resources(self, risk_level: str) -> List[Dict]:
        all_resources = [
            {"type": "rescue_team", "name": "NDRF Team Alpha", "personnel": 25, "icon": "🚁"},
            {"type": "rescue_team", "name": "Fire Brigade Unit", "personnel": 15, "icon": "🚒"},
            {"type": "medical", "name": "Mobile Hospital", "beds": 20, "icon": "🏥"},
            {"type": "equipment", "name": "Water Pumps", "capacity": "50000L/hr", "icon": "⚙️"},
            {"type": "shelter", "name": "Community Center", "capacity": 500, "icon": "🏛️"},
            {"type": "shelter", "name": "School Building", "capacity": 300, "icon": "🏫"}
        ]
        
        if risk_level == "HIGH":
            return all_resources
        elif risk_level == "MEDIUM":
            return all_resources[:3]
        return []
    
    def _get_safety_tips(self, risk_level: str) -> Dict[str, List[Dict]]:
        """Return contextual safety do's and don'ts based on risk level"""
        
        dos_high = [
            {"tip": "Move to higher ground immediately", "priority": "critical", "icon": "🏔️"},
            {"tip": "Keep emergency kit ready with essentials", "priority": "critical", "icon": "🎒"},
            {"tip": "Follow official evacuation orders", "priority": "critical", "icon": "🚨"},
            {"tip": "Keep mobile phones charged for emergency calls", "priority": "high", "icon": "📱"},
            {"tip": "Store important documents in waterproof bags", "priority": "high", "icon": "📄"},
            {"tip": "Stock clean drinking water (3L per person/day)", "priority": "high", "icon": "💧"},
            {"tip": "Monitor local news and disaster alerts continuously", "priority": "high", "icon": "📻"},
            {"tip": "Inform family members about your location", "priority": "medium", "icon": "👨‍👩‍👧‍👦"},
        ]
        
        donts_high = [
            {"tip": "Do NOT attempt to cross flooded roads or bridges", "priority": "critical", "icon": "🚫"},
            {"tip": "Do NOT ignore evacuation warnings", "priority": "critical", "icon": "⚠️"},
            {"tip": "Do NOT touch electrical equipment if wet", "priority": "critical", "icon": "⚡"},
            {"tip": "Do NOT drive through standing water", "priority": "high", "icon": "🚗"},
            {"tip": "Do NOT go near riverbanks or drainage channels", "priority": "high", "icon": "🌊"},
            {"tip": "Do NOT consume contaminated flood water", "priority": "high", "icon": "🚱"},
        ]
        
        dos_medium = [
            {"tip": "Stay alert and monitor weather updates", "priority": "high", "icon": "📡"},
            {"tip": "Prepare an emergency kit with food and water", "priority": "high", "icon": "🎒"},
            {"tip": "Identify safe evacuation routes in advance", "priority": "medium", "icon": "🗺️"},
            {"tip": "Clear drainage systems around your area", "priority": "medium", "icon": "🔧"},
            {"tip": "Keep vehicles fueled for possible evacuation", "priority": "medium", "icon": "⛽"},
            {"tip": "Charge all electronic devices", "priority": "medium", "icon": "🔋"},
        ]
        
        donts_medium = [
            {"tip": "Do NOT ignore weather warnings", "priority": "high", "icon": "⚠️"},
            {"tip": "Do NOT leave emergency supplies unpacked", "priority": "medium", "icon": "📦"},
            {"tip": "Do NOT block drainage pathways", "priority": "medium", "icon": "🚧"},
            {"tip": "Do NOT plan unnecessary outdoor activities", "priority": "medium", "icon": "🏕️"},
        ]
        
        dos_low = [
            {"tip": "Stay informed about weather forecasts", "priority": "low", "icon": "🌤️"},
            {"tip": "Keep emergency numbers saved", "priority": "low", "icon": "📞"},
            {"tip": "Review your flood insurance coverage", "priority": "low", "icon": "📋"},
        ]
        
        donts_low = [
            {"tip": "Do NOT ignore long-term flood risk areas", "priority": "low", "icon": "📍"},
            {"tip": "Do NOT dispose waste in water bodies", "priority": "low", "icon": "♻️"},
        ]
        
        if risk_level == "HIGH":
            return {"dos": dos_high, "donts": donts_high}
        elif risk_level == "MEDIUM":
            return {"dos": dos_medium, "donts": donts_medium}
        return {"dos": dos_low, "donts": donts_low}
    
    def _get_emergency_contacts(self) -> List[Dict]:
        return [
            {"name": "National Disaster Response Force", "number": "011-24363260", "icon": "🚁"},
            {"name": "Police Emergency", "number": "100", "icon": "🚔"},
            {"name": "Ambulance", "number": "108", "icon": "🚑"},
            {"name": "Fire Brigade", "number": "101", "icon": "🚒"},
            {"name": "Disaster Helpline", "number": "1078", "icon": "☎️"},
            {"name": "Women Helpline", "number": "1091", "icon": "👩"},
        ]
    
    def _get_evacuation_info(self, risk_level: str, location: str) -> Dict[str, Any]:
        if risk_level == "HIGH":
            return {
                "status": "IMMEDIATE EVACUATION RECOMMENDED",
                "urgency": "critical",
                "routes": [
                    {"direction": "North", "destination": "Higher Ground - Community Center", "distance": "2.5 km", "status": "Open"},
                    {"direction": "East", "destination": "Government School Shelter", "distance": "3.1 km", "status": "Open"},
                    {"direction": "West", "destination": "District Hospital", "distance": "4.0 km", "status": "Open"},
                ],
                "shelter_capacity": 800,
                "people_evacuated": 0
            }
        elif risk_level == "MEDIUM":
            return {
                "status": "PREPARE FOR POSSIBLE EVACUATION",
                "urgency": "moderate",
                "routes": [
                    {"direction": "North", "destination": "Community Center", "distance": "2.5 km", "status": "Standby"},
                ],
                "shelter_capacity": 500,
                "people_evacuated": 0
            }
        return {
            "status": "NO EVACUATION NEEDED",
            "urgency": "none",
            "routes": [],
            "shelter_capacity": 0,
            "people_evacuated": 0
        }
