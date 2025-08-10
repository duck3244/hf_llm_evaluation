import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    """프로젝트 설정 클래스"""

    # API 설정
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    HUGGINGFACE_API_BASE = "https://huggingface.co/api"

    # 데이터베이스 설정
    DATABASE_PATH = "data/llm_evaluations.db"

    # 디렉토리 설정
    REPORTS_DIR = "reports"
    EXPORTS_DIR = "exports"

    # 수집할 태스크 목록
    TASKS_TO_COLLECT = [
        "text-generation",
        "text-classification",
        "question-answering",
        "summarization",
        "translation",
        "token-classification",
        "fill-mask",
        "text2text-generation"
    ]

    # 태스크별 설정
    MODELS_PER_TASK = 30
    MAX_EVALUATIONS_PER_MODEL = 10

    # API 요청 설정
    API_DELAY = 0.1  # 초
    REQUEST_TIMEOUT = 30  # 초
    MAX_RETRIES = 3

    # 태스크 카테고리 정의
    TASK_CATEGORIES = {
        "text-generation": {
            "description": "텍스트 생성 태스크",
            "common_datasets": ["hellaswag", "arc", "mmlu", "truthfulqa", "winogrande"],
            "common_metrics": ["accuracy", "perplexity", "bleu", "rouge"]
        },
        "text-classification": {
            "description": "텍스트 분류 태스크",
            "common_datasets": ["glue", "imdb", "sst2", "rotten_tomatoes"],
            "common_metrics": ["accuracy", "f1", "precision", "recall"]
        },
        "question-answering": {
            "description": "질문 답변 태스크",
            "common_datasets": ["squad", "squad_v2", "natural_questions", "ms_marco"],
            "common_metrics": ["exact_match", "f1", "accuracy"]
        },
        "summarization": {
            "description": "요약 태스크",
            "common_datasets": ["cnn_dailymail", "xsum", "reddit_tifu", "newsroom"],
            "common_metrics": ["rouge-1", "rouge-2", "rouge-l", "bleu", "meteor"]
        },
        "translation": {
            "description": "번역 태스크",
            "common_datasets": ["wmt14", "wmt16", "opus", "multi30k"],
            "common_metrics": ["bleu", "meteor", "ter", "chrf"]
        },
        "token-classification": {
            "description": "토큰 분류 태스크 (NER 등)",
            "common_datasets": ["conll2003", "ontonotes5", "wikiann"],
            "common_metrics": ["f1", "precision", "recall", "accuracy"]
        },
        "fill-mask": {
            "description": "마스크 채우기 태스크",
            "common_datasets": ["lambada", "wikitext", "bookcorpus"],
            "common_metrics": ["accuracy", "perplexity", "top-k-accuracy"]
        }
    }

    # 로깅 설정
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def validate(cls):
        """설정 검증"""
        required_dirs = [cls.REPORTS_DIR, cls.EXPORTS_DIR, "data"]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_task_info(cls, task_name: str) -> Dict[str, Any]:
        """특정 태스크의 정보 반환"""
        return cls.TASK_CATEGORIES.get(task_name, {})