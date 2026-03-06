"""
agents/agent.py
Base Agent class — Ollama only version.
"""

import os
import requests
from typing import Optional
from agents.roles import RoleConfig


class Agent:

    def __init__(
        self,
        role: RoleConfig,
        model: str = "llama3.1",
        backend: str = "ollama",
        temperature: float = 0.7,
        agent_id: Optional[str] = None,
    ):
        self.role = role
        self.model = model
        self.backend = backend
        self.temperature = temperature
        self.agent_id = agent_id or f"{role.name.lower()}_{id(self) % 10000}"
        self.message_history = []
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    def send(self, user_message: str, context: Optional[str] = None) -> str:
        """Send a message to the agent and return its response."""
        full_system = self.role.system_prompt
        if context:
            full_system += f"\n\n--- Shared Context ---\n{context}"

        self.message_history.append({"role": "user", "content": user_message})

        response = self._call_ollama(full_system)

        self.message_history.append({"role": "assistant", "content": response})
        return response

    def _call_ollama(self, system: str) -> str:
        messages = [{"role": "system", "content": system}] + self.message_history
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def reset(self):
        """Clear conversation history between tasks."""
        self.message_history = []

    def __repr__(self):
        return f"Agent(id={self.agent_id}, role={self.role.name}, model={self.model})"