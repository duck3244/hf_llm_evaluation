"""
데이터 모델 정의 모듈
HuggingFace LLM 평가 데이터를 위한 데이터 클래스들 (수정된 버전)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import math


@dataclass
class ModelInfo:
    """모델 정보를 저장하는 데이터 클래스"""
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

    def __post_init__(self):
        """데이터 검증 및 정규화"""
        # 음수 값 방지
        self.downloads = max(0, self.downloads)
        self.likes = max(0, self.likes)

        # 빈 문자열을 None으로 변환
        if self.model_name == "":
            self.model_name = self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id

        # 태그 중복 제거
        self.tags = list(set(self.tags)) if self.tags else []
        self.task_categories = list(set(self.task_categories)) if self.task_categories else []

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """딕셔너리에서 객체 생성 (안전한 변환)"""
        # 필수 필드 검증
        if 'model_id' not in data or not data['model_id']:
            raise ValueError("model_id는 필수 필드입니다.")

        # 기본값 설정
        safe_data = {
            'model_id': data['model_id'],
            'model_name': data.get('model_name', data['model_id'].split('/')[-1]),
            'author': data.get('author', ''),
            'downloads': max(0, data.get('downloads', 0)),
            'likes': max(0, data.get('likes', 0)),
            'created_at': data.get('created_at'),
            'last_modified': data.get('last_modified'),
            'library_name': data.get('library_name'),
            'pipeline_tag': data.get('pipeline_tag'),
            'tags': data.get('tags', []),
            'task_categories': data.get('task_categories', []),
            'model_size': data.get('model_size'),
            'license': data.get('license')
        }

        return cls(**safe_data)

    def __str__(self) -> str:
        return f"ModelInfo(id={self.model_id}, downloads={self.downloads:,})"


@dataclass
class EvaluationResult:
    """평가 결과를 저장하는 데이터 클래스"""
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

    def __post_init__(self):
        """데이터 검증 및 정규화"""
        # NaN 값 체크 및 변환
        if isinstance(self.metric_value, str):
            try:
                self.metric_value = float(self.metric_value)
            except (ValueError, TypeError):
                raise ValueError(f"잘못된 메트릭 값: {self.metric_value}")

        # NaN, Infinity 체크
        if not isinstance(self.metric_value, (int, float)) or math.isnan(self.metric_value) or math.isinf(self.metric_value):
            raise ValueError(f"유효하지 않은 메트릭 값: {self.metric_value}")

        # 기본값 설정
        if self.metric_config is None:
            self.metric_config = {}
        if self.additional_info is None:
            self.additional_info = {}

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """딕셔너리에서 객체 생성 (안전한 변환)"""
        # 필수 필드 검증
        required_fields = ['model_id', 'dataset_name', 'metric_name', 'metric_value', 'task_type']
        for field in required_fields:
            if field not in data or data[field] is None:
                raise ValueError(f"{field}는 필수 필드입니다.")

        return cls(**data)

    def __str__(self) -> str:
        return f"EvaluationResult({self.model_id}, {self.metric_name}={self.metric_value:.4f})"


@dataclass
class TaskCategory:
    """태스크 카테고리 정보를 저장하는 데이터 클래스"""
    task_name: str
    description: str
    common_datasets: List[str] = field(default_factory=list)
    common_metrics: List[str] = field(default_factory=list)
    subcategories: List[str] = field(default_factory=list)

    def __post_init__(self):
        """데이터 검증"""
        if not self.task_name or not self.task_name.strip():
            raise ValueError("task_name은 필수 필드입니다.")
        if not self.description or not self.description.strip():
            raise ValueError("description은 필수 필드입니다.")

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

    def __post_init__(self):
        """데이터 검증"""
        if not self.dataset_name or not self.dataset_name.strip():
            raise ValueError("dataset_name은 필수 필드입니다.")
        if not self.task_type or not self.task_type.strip():
            raise ValueError("task_type은 필수 필드입니다.")

        # 크기는 양수여야 함
        if self.size is not None and self.size < 0:
            self.size = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """딕셔너리에서 객체 생성"""
        return cls(**data)

    def __str__(self) -> str:
        return f"DatasetInfo({self.dataset_name}, {self.task_type})"


@dataclass
class CollectionStats:
    """데이터 수집 통계를 저장하는 데이터 클래스"""
    collection_date: str = field(default_factory=lambda: datetime.now().isoformat())
    total_models: int = 0
    total_evaluations: int = 0
    tasks_collected: List[str] = field(default_factory=list)
    collection_duration: Optional[float] = None
    errors_count: int = 0
    success_rate: float = 0.0

    def __post_init__(self):
        """데이터 검증"""
        # 음수 값 방지
        self.total_models = max(0, self.total_models)
        self.total_evaluations = max(0, self.total_evaluations)
        self.errors_count = max(0, self.errors_count)

        # 성공률 범위 제한
        self.success_rate = max(0.0, min(1.0, self.success_rate))

        # 중복 태스크 제거
        self.tasks_collected = list(set(self.tasks_collected)) if self.tasks_collected else []

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    def update_success_rate(self):
        """성공률 업데이트"""
        total_attempts = self.total_models + self.errors_count
        if total_attempts > 0:
            self.success_rate = self.total_models / total_attempts
        else:
            self.success_rate = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollectionStats':
        """딕셔너리에서 객체 생성"""
        return cls(**data)

    def __str__(self) -> str:
        return (f"CollectionStats(models={self.total_models}, "
                f"evaluations={self.total_evaluations}, "
                f"tasks={len(self.tasks_collected)}, "
                f"success_rate={self.success_rate:.2%})")


# 유틸리티 함수들 (개선된 버전)
def serialize_for_db(obj: Any) -> str:
    """데이터베이스 저장을 위한 안전한 직렬화"""
    if obj is None:
        return ""

    if isinstance(obj, (list, dict)):
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            # 직렬화 실패 시 문자열로 변환
            return str(obj)

    return str(obj)


def deserialize_from_db(data: str, target_type: type = list) -> Any:
    """데이터베이스에서 안전한 역직렬화"""
    if not data or data.strip() == "":
        return [] if target_type == list else {} if target_type == dict else None

    try:
        if target_type in (list, dict):
            result = json.loads(data)
            # 타입 검증
            if not isinstance(result, target_type):
                return [] if target_type == list else {}
            return result
        return data
    except (json.JSONDecodeError, TypeError, ValueError):
        # 파싱 실패 시 기본값 반환
        return [] if target_type == list else {} if target_type == dict else data


# 검증 함수들 (개선된 버전)
def validate_model_info(model_info: ModelInfo) -> bool:
    """모델 정보 유효성 검증"""
    try:
        if not model_info.model_id or not model_info.model_id.strip():
            return False
        if not model_info.model_name or not model_info.model_name.strip():
            return False
        if model_info.downloads < 0:
            return False
        if model_info.likes < 0:
            return False
        return True
    except Exception:
        return False


def validate_evaluation_result(evaluation: EvaluationResult) -> bool:
    """평가 결과 유효성 검증"""
    try:
        if not evaluation.model_id or not evaluation.model_id.strip():
            return False
        if not evaluation.dataset_name or not evaluation.dataset_name.strip():
            return False
        if not evaluation.metric_name or not evaluation.metric_name.strip():
            return False
        if not isinstance(evaluation.metric_value, (int, float)):
            return False
        # NaN, Infinity 체크
        if math.isnan(evaluation.metric_value) or math.isinf(evaluation.metric_value):
            return False
        return True
    except Exception:
        return False


def validate_task_category(task_category: TaskCategory) -> bool:
    """태스크 카테고리 유효성 검증"""
    try:
        if not task_category.task_name or not task_category.task_name.strip():
            return False
        if not task_category.description or not task_category.description.strip():
            return False
        return True
    except Exception:
        return False


def validate_dataset_info(dataset_info: DatasetInfo) -> bool:
    """데이터셋 정보 유효성 검증"""
    try:
        if not dataset_info.dataset_name or not dataset_info.dataset_name.strip():
            return False
        if not dataset_info.task_type or not dataset_info.task_type.strip():
            return False
        if dataset_info.size is not None and dataset_info.size < 0:
            return False
        return True
    except Exception:
        return False


# 타입 별칭
ModelList = List[ModelInfo]
EvaluationList = List[EvaluationResult]
TaskCategoryDict = Dict[str, TaskCategory]
DatasetDict = Dict[str, DatasetInfo]


# 팩토리 함수들
def create_model_info_from_api(api_data: Dict[str, Any], task: Optional[str] = None) -> ModelInfo:
    """API 응답에서 ModelInfo 객체 생성"""
    task_categories = []
    if task:
        task_categories.append(task)

    # 태그에서 추가 태스크 추출
    tags = api_data.get('tags', [])

    return ModelInfo(
        model_id=api_data.get('id', ''),
        model_name=api_data.get('id', '').split('/')[-1] if api_data.get('id') else '',
        author=api_data.get('author', ''),
        downloads=api_data.get('downloads', 0),
        likes=api_data.get('likes', 0),
        created_at=api_data.get('createdAt'),
        last_modified=api_data.get('lastModified'),
        library_name=api_data.get('library_name'),
        pipeline_tag=api_data.get('pipeline_tag'),
        tags=tags,
        task_categories=task_categories,
        model_size=extract_model_size(api_data),
        license=api_data.get('cardData', {}).get('license') if api_data.get('cardData') else None
    )


def extract_model_size(model_data: Dict[str, Any]) -> Optional[str]:
    """모델 크기 정보 추출"""
    size_patterns = ['7b', '13b', '30b', '65b', '70b', '175b', 'small', 'base', 'large', 'xl']

    # 태그에서 검색
    tags = model_data.get('tags', [])
    for tag in tags:
        tag_lower = tag.lower()
        for pattern in size_patterns:
            if pattern in tag_lower:
                return tag

    # 모델 ID에서 검색
    model_id = model_data.get('id', '').lower()
    for pattern in size_patterns:
        if pattern in model_id:
            return pattern.upper()

    return None