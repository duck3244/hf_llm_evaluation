"""
HuggingFace LLM 평가 데이터 수집 프로젝트
"""

__version__ = "1.0.0"
__author__ = "HF LLM Evaluation Team"
__description__ = "HuggingFace LLM 성능 평가 데이터 수집 및 분석 도구"

from .collectors.evaluation_collector import LLMEvaluationCollector
from .models.data_models import ModelInfo, EvaluationResult, TaskCategory
from .api.huggingface_api import HuggingFaceAPI
from .database.db_manager import DatabaseManager

__all__ = [
    "LLMEvaluationCollector",
    "ModelInfo",
    "EvaluationResult",
    "TaskCategory",
    "HuggingFaceAPI",
    "DatabaseManager"
]