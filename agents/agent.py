import os
import json
import requests
from typing import Any,Optional
from agents.roles import RoleConfig

class Agent:
    """
       A single LLM-backed agent with a defined role
       Supports: Ollama(local),openAI API,Anthropic API
    """
    def __int__(
            self,
            role:RoleConfig,
            model:str='llama3.1',
            backend:str='ollama',
            temperature:float=0.7,
            agent_id:Optional[str]=None
    ):

        self.role=role
        self.model=model
        self.backend=backend
        self.temperature=temperature
        self.agent_id=agent_id or f'{role.name.lower()}_{id(self)%10000}'
        self.message_history:list[dict]=[]
        self._setup_client()

def    