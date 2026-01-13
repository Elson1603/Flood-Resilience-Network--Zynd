from agents.base_agent import BaseAgent
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import os

class FloodMLP(nn.Module):
    """Improved Multi-Layer Perceptron for flood prediction"""
    def __init__(self, input_dim):
        super(FloodMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class PredictionAgent(BaseAgent):
    def __init__(self, agent_id: str, model_path: str):
        capabilities = ["flood_prediction", "risk_assessment", "ml_inference"]
        super().__init__(agent_id, "PredictionAgent", capabilities)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.feature_columns = checkpoint.get('feature_columns', 
                                             ['elevation_m', 'rainfall_mm', 'river_proximity', 'slope_deg'])
        input_dim = len(self.feature_columns)
        model_type = checkpoint.get('model_type', 'LSTM')
        
        # Load appropriate model architecture
        if model_type == 'MLP':
            self.model = FloodMLP(input_dim=input_dim).to(self.device)
        else:
            # Fallback to LSTM for older models
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from models.train_model import FloodLSTM
            self.model = FloodLSTM(input_dim=input_dim).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler = checkpoint['scaler']
        self.model_type = model_type
        
        print(f"   ✓ Model loaded: {model_type} with {input_dim} features, F1={checkpoint.get('f1_score', 0):.3f}")
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import datetime
        self.status = "predicting"
        self.last_action = datetime.datetime.now()
        self.action_count += 1
        
        print(f"\n🤖 PREDICTION AGENT")
        
        try:
            # Extract or estimate all 20 features
            # Get basic inputs from frontend
            elevation_m = data.get('elevation_m', 50)
            rainfall_mm = data.get('rainfall_mm', 50)
            slope_deg = data.get('slope_deg', 5)
            river_proximity_binary = data.get('river_proximity', 1)
            
            # Convert binary river proximity to distance (0=far, 1=near)
            distance_to_water_m = 100 if river_proximity_binary == 1 else 2000
            
            # Generate realistic estimates for missing features based on inputs
            features_dict = {
                'elevation_m': elevation_m,
                'slope_deg': slope_deg,
                'aspect_deg': 180,  # Default south-facing
                'curvature': -0.5 if elevation_m < 100 else 0.5,  # Low elevation = concave
                'flow_accumulation': max(1000, 10000 / (elevation_m + 1)),
                'distance_to_water_m': distance_to_water_m,
                'watershed_area_km2': 50,
                'drainage_density': 2.0,
                'soil_permeability_mm_hr': 15,
                'topographic_wetness_index': 12 if elevation_m < 100 else 8,
                'rainfall_24h_mm': rainfall_mm,
                'antecedent_rainfall_7d_mm': rainfall_mm * 0.6,
                'soil_moisture_pct': min(80, rainfall_mm * 0.15),
                'temperature_c': 25,
                'urban_density_pct': 40,
                'vegetation_cover_pct': 50,
                'impervious_surface_pct': 30,
                'drainage_capacity': 0.6,
                'season': 2,  # Spring
                'month': 6    # June
            }
            
            # Create feature array in correct order
            if len(self.feature_columns) == 20:
                features = np.array([[features_dict[col] for col in self.feature_columns]])
            else:
                # Fallback to old 4-feature model
                features = np.array([[elevation_m, rainfall_mm, river_proximity_binary, slope_deg]])
            
            print(f"   Features ({len(self.feature_columns)}): elevation={elevation_m}m, rainfall={rainfall_mm}mm, slope={slope_deg}°")
            
            # Normalize and predict
            features_scaled = self.scaler.transform(features)
            
            if self.model_type == 'MLP':
                # MLP takes simple 2D input
                input_tensor = torch.FloatTensor(features_scaled).to(self.device)
            else:
                # LSTM requires sequence input
                sequence = np.repeat(features_scaled, 10, axis=0)
                input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_tensor)
                flood_probability = prediction.item()
            
            risk_level = self._categorize_risk(flood_probability)
            
            print(f"   ✓ Prediction: {flood_probability*100:.1f}% | Risk: {risk_level}")
            
            self.status = "idle"
            
            return {
                "location": data.get('location', {}),
                "flood_probability": flood_probability,
                "risk_level": risk_level,
                "timestamp": data.get('timestamp'),
                "features_used": {
                    "elevation_m": elevation_m,
                    "rainfall_24h_mm": rainfall_mm,
                    "slope_deg": slope_deg,
                    "distance_to_water_m": distance_to_water_m
                }
            }
            
        except Exception as e:
            self.status = "error"
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _categorize_risk(self, probability: float) -> str:
        if probability >= 0.65:
            return "HIGH"
        elif probability >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"
