#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, logging, re, traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from medical_ai_agents.tools.base_tools import BaseTool

class ThoughtType(str, Enum):
    INITIAL = "initial"
    REASONING = "reasoning"
    OBSERVATION = "observation"
    CONCLUSION = "conclusion"

@dataclass
class ReActStep:
    thought: str
    thought_type: ThoughtType
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: str = datetime.now().isoformat()

class ReActCallbackHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str):
        self.logger = logging.getLogger(f"react.{agent_name}")
    def on_llm_start(self, *_, **__):
        self.logger.debug("LLM start")
    def on_llm_end(self, *_, **__):
        self.logger.debug("LLM end")

class BaseAgent(ABC):
    def __init__(self, name: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        self.name = name
        self.device = device
        self.logger = logging.getLogger(f"agent.{self.name.lower().replace(' ', '_')}")
        self.initialized = False
        self.callback_handler = ReActCallbackHandler(self.name)
        self.llm = ChatOpenAI(model=llm_model, temperature=0.5, callbacks=[self.callback_handler])
        self.tools: List[BaseTool] = self._register_tools()
        self.tool_descriptions = self._get_tool_descriptions()
        self.max_iterations = 10
        self.react_history: List[ReActStep] = []

    @abstractmethod
    def _register_tools(self) -> List[BaseTool]:
        ...
    @abstractmethod
    def _get_agent_description(self) -> str:
        ...
    @abstractmethod
    def initialize(self) -> bool:
        ...
    @abstractmethod
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...
    @abstractmethod
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        ...
    @abstractmethod
    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def _get_system_prompt(self) -> str:
        return (
            f"You are {self.name}, an expert medical AI agent using the ReAct pattern.\n\n"
            f"{self._get_agent_description()}\n\n"
            f"Available tools:\n{self.tool_descriptions}\n\n"
            "Rules:\n"
            "1. Begin each step with 'Thought:' then 'Action:' then 'Action Input:' if needed.\n"
            "2. Use tool name exactly or 'Final Answer'.\n"
            "3. Continue until action is 'Final Answer'."
        )

    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        try:
            data = json.loads(synthesis)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"answer": synthesis}

    def _get_tool_descriptions(self) -> str:
        if not self.tools:
            return "None"
        out = []
        for t in self.tools:
            schema = t.get_parameters_schema()
            params = [f"  - {p} ({meta.get('type','any')})" for p, meta in schema.items()]
            out.append(f"- {t.name}:\n" + "\n".join(params))
        return "\n".join(out)

    def _parse_llm_response(self, txt: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        thought = re.search(r"Thought:\s*(.+?)(?=Action:|$)", txt, re.DOTALL)
        action = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", txt, re.DOTALL)
        a_input = re.search(r"Action Input:\s*(\{.+?\})", txt, re.DOTALL)
        thought_val = thought.group(1).strip() if thought else None
        action_val = action.group(1).strip() if action else None
        try:
            input_val = json.loads(a_input.group(1)) if a_input else None
        except Exception:
            input_val = None
        return thought_val, action_val, input_val

    def _execute_tool(self, name: str, params: Dict[str, Any]) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if not tool:
            return f"Error: tool '{name}' not found."
        try:
            res = tool(**params)
            return json.dumps(res, ensure_ascii=False, indent=2) if isinstance(res, dict) else str(res)
        except Exception as e:
            return f"Error executing {name}: {e}"

    def _create_react_messages(self, task_input: Dict[str, Any]) -> List[Any]:
        msgs = [SystemMessage(content=self._get_system_prompt()), HumanMessage(content=self._format_task_input(task_input))]
        if self.react_history:
            hist = []
            for s in self.react_history[-5:]:
                hist.append(f"Thought: {s.thought}\nAction: {s.action}\nAction Input: {json.dumps(s.action_input)}\nObservation: {s.observation[:120] if s.observation else ''}")
            msgs.append(AIMessage(content="\n".join(hist)))
        return msgs

    def _run_react_loop(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run ReAct loop for guided adaptive classification."""
        self.react_history = []
        for i in range(1, self.max_iterations + 1):
            resp = self.llm.invoke(self._create_react_messages(task_input)).content
            print(f"🔧 DEBUG: resp {i} {resp}")
            t, a, inp = self._parse_llm_response(resp)
            print(f"🔧 DEBUG: t {i} {t}")
            print(f"🔧 DEBUG: a {i} {a}")
            print(f"🔧 DEBUG: inp {i} {inp}")
            if not t or not a:
                continue
            step = ReActStep(thought=t, thought_type=ThoughtType.INITIAL if i == 1 else ThoughtType.REASONING, action=a, action_input=inp)
            if a.lower() == "final answer":
                step.thought_type = ThoughtType.CONCLUSION
                self.react_history.append(step)
                answer = inp.get("answer") if inp else t
                return {"success": True, "answer": answer, "history": self._serialize_history()}
            obs = self._execute_tool(a, inp or {})
            step.observation = obs
            step.thought_type = ThoughtType.OBSERVATION
            self.react_history.append(step)
            task_input[f"obs_{i}"] = obs
            print(f"🔧 DEBUG: ReAct iteration {i}")
            print(f"🔧 DEBUG: Thought: {t}")
            print(f"🔧 DEBUG: Action: {a}")
            print(f"🔧 DEBUG: Action Input: {inp}")
            print(f"🔧 DEBUG: Observation: {obs}")
        return {"success": False, "error": "Max iterations reached", "history": self._serialize_history()}

    def _serialize_history(self) -> List[Dict[str, Any]]:
        return [{"thought": s.thought, "action": s.action, "observation": s.observation} for s in self.react_history]

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.initialized:
                self.initialized = self.initialize()
                if not self.initialized:
                    return {**state, "error": f"Init {self.name} failed"}
            task_input = self._extract_task_input(state)
            result = self._run_react_loop(task_input)
            agent_out = self._format_agent_result(result)
            print(f"🔧 DEBUG: agent_out {agent_out}")
            return {**state, **agent_out}
        except Exception as e:
            err = f"Error in {self.name}: {e}\n{traceback.format_exc()}"
            self.logger.error(err)
            return {**state, "error": err}

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(state)
