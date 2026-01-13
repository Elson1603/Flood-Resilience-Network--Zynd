from agents.base_agent import BaseAgent
from typing import Dict, Any
import asyncio

class CoordinationAgent(BaseAgent):
    def __init__(self, agent_id: str, agents: Dict):
        capabilities = ["workflow_coordination", "decision_making", "orchestration"]
        super().__init__(agent_id, "CoordinationAgent", capabilities)
        self.agents = agents
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.status = "coordinating"
        
        print(f"\n{'='*80}")
        print("⚙️  COORDINATION WORKFLOW STARTED")
        print(f"{'='*80}")
        
        # Step 1: Monitor
        monitoring_result = await self.agents['monitoring'].execute(data)
        
        # Step 2: Predict
        prediction_result = await self.agents['prediction'].execute(monitoring_result)
        
        # Step 3: Alert & Mobilize if needed
        risk_level = prediction_result.get('risk_level')
        
        if risk_level in ['HIGH', 'MEDIUM']:
            await asyncio.gather(
                self.agents['alert'].execute(prediction_result),
                self.agents['resource'].execute(prediction_result)
            )
        
        print(f"{'='*80}")
        print(f"✓ WORKFLOW COMPLETE - {risk_level} RISK")
        print(f"{'='*80}\n")
        
        self.status = "idle"
        return prediction_result
