from agents.base_agent import BaseAgent
from typing import Dict, Any
import datetime
import random
import math

class MonitoringAgent(BaseAgent):
    def __init__(self, agent_id: str):
        capabilities = ["weather_monitoring", "data_collection", "real_time_feeds", "weather_simulation"]
        super().__init__(agent_id, "MonitoringAgent", capabilities)
    
    def get_simulated_weather(self, lat: float = 19.076, lon: float = 72.877, rainfall_mm: float = 0) -> Dict[str, Any]:
        """Generate realistic simulated weather conditions based on location and rainfall"""
        # Base temperature varies by latitude
        base_temp = 35 - abs(lat - 23.5) * 0.5  # Hotter near tropics
        temp = base_temp + random.uniform(-3, 3)
        
        # Humidity correlates with rainfall
        base_humidity = min(95, 40 + rainfall_mm * 0.08)
        humidity = base_humidity + random.uniform(-5, 5)
        
        # Wind speed increases with rainfall intensity
        wind_speed = 5 + (rainfall_mm / 100) * 15 + random.uniform(-2, 3)
        
        # Cloud cover correlates with rainfall
        cloud_cover = min(100, 20 + rainfall_mm * 0.1)
        
        # Visibility decreases with heavy rain
        visibility = max(0.5, 10 - rainfall_mm * 0.01)
        
        # Pressure drops during storms
        pressure = 1013 - (rainfall_mm / 100) * 8 + random.uniform(-2, 2)
        
        # UV index
        uv_index = max(0, 8 - cloud_cover * 0.07 + random.uniform(-1, 1))
        
        return {
            "temperature_c": round(temp, 1),
            "feels_like_c": round(temp + humidity * 0.05, 1),
            "humidity_pct": round(min(100, max(20, humidity)), 1),
            "wind_speed_kmh": round(max(0, wind_speed), 1),
            "wind_direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            "cloud_cover_pct": round(min(100, max(0, cloud_cover)), 1),
            "visibility_km": round(max(0.1, visibility), 1),
            "pressure_hpa": round(pressure, 1),
            "uv_index": round(max(0, uv_index), 1),
            "condition": self._get_weather_condition(rainfall_mm, cloud_cover),
            "condition_icon": self._get_weather_icon(rainfall_mm, cloud_cover),
        }
    
    def _get_weather_condition(self, rainfall: float, cloud_cover: float) -> str:
        if rainfall > 500:
            return "Extreme Rainfall"
        elif rainfall > 200:
            return "Heavy Thunderstorm"
        elif rainfall > 100:
            return "Heavy Rain"
        elif rainfall > 50:
            return "Moderate Rain"
        elif rainfall > 10:
            return "Light Rain"
        elif cloud_cover > 70:
            return "Overcast"
        elif cloud_cover > 40:
            return "Partly Cloudy"
        else:
            return "Clear Sky"
    
    def _get_weather_icon(self, rainfall: float, cloud_cover: float) -> str:
        if rainfall > 200:
            return "⛈️"
        elif rainfall > 100:
            return "🌧️"
        elif rainfall > 50:
            return "🌦️"
        elif rainfall > 10:
            return "🌧️"
        elif cloud_cover > 70:
            return "☁️"
        elif cloud_cover > 40:
            return "⛅"
        else:
            return "☀️"
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.status = "monitoring"
        self.last_action = datetime.datetime.now()
        self.action_count += 1
        
        location = data.get("location", {})
        lat = location.get('lat', 19.076)
        lon = location.get('lon', 72.877)
        rainfall_mm = location.get('rainfall_mm', 0)
        
        print(f"\n🔍 MONITORING AGENT")
        print(f"   Location: {location.get('name', 'Unknown')} ({lat}, {lon})")
        
        # Get simulated weather
        weather = self.get_simulated_weather(lat, lon, rainfall_mm)
        
        # Use form input data directly
        processed_data = {
            "elevation_m": location.get('elevation_m', 0),
            "rainfall_mm": rainfall_mm,
            "river_proximity": location.get('river_proximity', 1),
            "slope_deg": location.get('slope_deg', 0),
            "location": location,
            "weather": weather,
            "timestamp": datetime.datetime.now().isoformat(),
            "data_source": "Form Input + Weather Simulation"
        }
        
        print(f"   ✓ Data collected: {processed_data['elevation_m']}m, {processed_data['rainfall_mm']}mm")
        print(f"   ✓ Weather: {weather['condition']} {weather['condition_icon']}, {weather['temperature_c']}°C, {weather['humidity_pct']}% humidity")
        
        self.status = "idle"
        return processed_data
