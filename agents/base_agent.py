from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime
import asyncio

class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = "ready"
        self.last_action = None
        self.action_count = 0
        
        print(f"✓ {agent_type} initialized: {agent_id}")
        print(f"  Capabilities: {', '.join(capabilities)}")
    
    @abstractmethod
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    async def report_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "capabilities": self.capabilities,
            "last_action": self.last_action.isoformat() if self.last_action else None,
            "action_count": self.action_count
        }
