"""
데이터 모델 정의 모듈
HuggingFace LLM 평가 데이터를 위한 데이터 클래스들 (README 예시와 일치)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json


@dataclass
class ModelInfo:
    """모델 정보를 저장하는 데이터 클래스 (README 예시 구현)"""
    model_id: str
    model_name: str
    author: str
    downloads: int = 0
    likes: int = 0
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    library_name: Optional[str] = None
    pipeline_tag: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    task_categories: List[str] = field(default_factory=list)
    model_size: Optional[str] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """딕셔너리에서 객체 생성"""
        return cls(**data)

    def __str__(self) -> str:
        return f"ModelInfo(id={self.model_id}, downloads={self.downloads:,})"


@dataclass
class EvaluationResult:
    """평가 결과를 저장하는 데이터 클래스 (README 예시 구현)"""
    model_id: str
    dataset_name: str
    metric_name: str
    metric_value: float
    task_type: str
    evaluation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_version: Optional[str] = None
    metric_config: Optional[Dict[str, Any]] = None
    additional_info: Optional[Dict[str, Any]] = None
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """딕셔너리에서 객체 생성"""
        return cls(**data)

    def __str__(self) -> str:
        return f"EvaluationResult({self.model_id}, {self.metric_name}={self.metric_value:.4f})"


@dataclass
class TaskCategory:
    """태스크 카테고리 정보를 저장하는 데이터 클래스 (README 태스크 테이블 구현)"""
    task_name: str
    description: str
    common_datasets: List[str] = field(default_factory=list)
    common_metrics: List[str] = field(default_factory=list)
    subcategories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskCategory':
        """딕셔너리에서 객체 생성"""
        return cls(**data)

    def __str__(self) -> str:
        return f"TaskCategory({self.task_name}: {self.description})"


@dataclass
class DatasetInfo:
    """데이터셋 정보를 저장하는 데이터 클래스"""
    dataset_name: str
    task_type: str
    description: Optional[str] = None
    size: Optional[int] = None
    language: Optional[str] = None
    splits: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """딕셔너리에서 객체 생성"""
        return cls(**data)


@dataclass
class CollectionStats:
    """데이터 수집 통계를 저장하는 데이터 클래스 (README 통계 형식 구현)"""
    collection_date: str = field(default_factory=lambda: datetime.now().isoformat())
    total_models: int = 0
    total_evaluations: int = 0
    tasks_collected: List[str] = field(default_factory=list)
    collection_duration: Optional[float] = None
    errors_count: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    def update_success_rate(self):
        """성공률 업데이트"""
        total_attempts = self.total_models + self.errors_count
        if total_attempts > 0:
            self.success_rate = self.total_models / total_attempts

    def __str__(self) -> str:
        return (f"CollectionStats(models={self.total_models}, "
                f"evaluations={self.total_evaluations}, "
                f"tasks={len(self.tasks_collected)}, "
                f"success_rate={self.success_rate:.2%})")


# 유틸리티 함수들
def serialize_for_db(obj: Any) -> str:
    """데이터베이스 저장을 위한 직렬화"""
    if isinstance(obj, (list, dict)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj) if obj is not None else ""


def deserialize_from_db(data: str, target_type: type = list) -> Any:
    """데이터베이스에서 역직렬화"""
    try:
        if not data:
            return [] if target_type == list else {}
        if target_type in (list, dict):
            return json.loads(data)
        return data
    except (json.JSONDecodeError, TypeError):
        return [] if target_type == list else {}


# 검증 함수들 (README의 데이터 품질 검사 구현)
def validate_model_info(model_info: ModelInfo) -> bool:
    """모델 정보 유효성 검증"""
    if not model_info.model_id or not model_info.model_id.strip():
        return False
    if not model_info.model_name or not model_info.model_name.strip():
        return False
    if model_info.downloads < 0:
        return False
    if model_info.likes < 0:
        return False
    return True


def validate_evaluation_result(evaluation: EvaluationResult) -> bool:
    """평가 결과 유효성 검증"""
    if not evaluation.model_id or not evaluation.model_id.strip():
        return False
    if not evaluation.dataset_name or not evaluation.dataset_name.strip():
        return False
    if not evaluation.metric_name or not evaluation.metric_name.strip():
        return False
    if not isinstance(evaluation.metric_value, (int, float)):
        return False
    # NaN 값 체크
    if evaluation.metric_value != evaluation.metric_value:  # NaN 체크
        return False
    return True


def validate_task_category(task_category: TaskCategory) -> bool:
    """태스크 카테고리 유효성 검증"""
    if not task_category.task_name or not task_category.task_name.strip():
        return False
    if not task_category.description or not task_category.description.strip():
        return False
    return True


# 타입 별칭 (README 예시에서 사용)
ModelList = List[ModelInfo]
EvaluationList = List[EvaluationResult]
TaskCategoryDict = Dict[str, TaskCategory]