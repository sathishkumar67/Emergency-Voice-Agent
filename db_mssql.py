from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


CallType = Literal["emergency", "inquiry", "report"]
Stage = Literal["need_language", "collecting", "confirming", "created", "ended"]


class ExtractionConfidence(BaseModel):
    overall: float = Field(ge=0.0, le=1.0, default=0.0)
    fields: Dict[str, float] = Field(default_factory=dict)  # e.g. {"location":0.7}


class CallInfo(BaseModel):
    call_type: Optional[CallType] = None
    intent: Optional[str] = None

    incident_type: Optional[str] = None
    location: Optional[str] = None
    caller_name: Optional[str] = None

    optional: Dict[str, str] = Field(default_factory=dict)
    confidence: ExtractionConfidence = Field(default_factory=ExtractionConfidence)

    confirmed: bool = False


class Metrics(BaseModel):
    call_id: str
    started_at_unix: float
    ended_at_unix: Optional[float] = None

    user_turns: int = 0
    agent_turns: int = 0

    stt_final_count: int = 0
    stt_interim_count: int = 0

    extraction_updates: int = 0
    incident_created: bool = False
    incident_id: Optional[int] = None


class CallState(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    language: str = "en"
    stage: Stage = "need_language"

    info: CallInfo = Field(default_factory=CallInfo)
    metrics: Metrics = Field(default_factory=lambda: Metrics(call_id=str(uuid.uuid4()), started_at_unix=time.time()))

    def model_post_init(self, __context: Any) -> None:
        # ensure metrics.call_id matches
        self.metrics.call_id = self.call_id
        if not self.metrics.started_at_unix:
            self.metrics.started_at_unix = time.time()

    def required_missing(self) -> list[str]:
        """
        Define required fields for your workflow.
        - For emergency: incident_type + location + caller_name
        - For inquiry/report: location optional, name still desired
        """
        missing = []
        if self.info.call_type is None:
            missing.append("call_type")
        if self.info.incident_type is None:
            missing.append("incident_type")
        if self.info.location is None:
            missing.append("location")
        if self.info.caller_name is None:
            missing.append("caller_name")
        return missing