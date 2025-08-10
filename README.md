# HuggingFace LLM 성능 평가 데이터 수집 프로젝트

HuggingFace Hub에서 LLM 모델들의 성능 평가 데이터를 자동으로 수집하고 분석하는 종합적인 도구입니다.

## 📋 주요 기능

- **자동 데이터 수집**: HuggingFace API를 통한 모델 정보 및 평가 결과 수집
- **태스크별 분류**: 텍스트 생성, 분류, 질문 답변 등 다양한 태스크별 모델 관리
- **성능 분석**: 태스크별 리더보드 및 상세 성능 리포트 생성
- **데이터 저장**: SQLite를 사용한 효율적인 데이터 관리
- **내보내기**: CSV 형태로 데이터 내보내기 지원

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/hf-llm-evaluation.git
cd hf-llm-evaluation

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (선택사항)
cp .env.example .env
# .env 파일에서 HUGGINGFACE_TOKEN 설정
```

### 2. 기본 사용법

```bash
# 현재 수집 현황 확인
python main.py --stats

# 모든 태스크 데이터 수집
python main.py --collect-all

# 특정 태스크만 수집
python main.py --task text-generation

# 리포트 생성
python main.py --reports

# 데이터 내보내기
python main.py --export
```

### 3. 프로그래밍 방식 사용

```python
from src.collectors.evaluation_collector import LLMEvaluationCollector

# 수집기 초기화
collector = LLMEvaluationCollector()

# 특정 태스크 모델 수집
models = collector.collect_models_by_task("text-generation", limit=10)

# 리더보드 생성
leaderboard = collector.db_manager.get_task_leaderboard(
    task_type="text-generation", 
    metric_name="accuracy"
)

print(leaderboard.head())
```

## 📊 지원하는 태스크

| 태스크 | 설명 | 주요 데이터셋 | 주요 메트릭 |
|--------|------|---------------|-------------|
| text-generation | 텍스트 생성 | hellaswag, arc, mmlu | accuracy, perplexity |
| text-classification | 텍스트 분류 | glue, imdb, sst2 | accuracy, f1 |
| question-answering | 질문 답변 | squad, natural_questions | exact_match, f1 |
| summarization | 요약 | cnn_dailymail, xsum | rouge-1, rouge-2 |
| translation | 번역 | wmt14, opus | bleu, meteor |

## 🗂️ 프로젝트 구조

```
hf_llm_evaluation/
├── README.md                    # 프로젝트 문서
├── requirements.txt             # Python 의존성
├── config.py                    # 설정 파일
├── main.py                      # 메인 실행 스크립트
├── src/                         # 소스 코드
│   ├── models/                  # 데이터 모델
│   │   └── data_models.py
│   ├── api/                     # API 클라이언트
│   │   └── huggingface_api.py
│   ├── database/                # 데이터베이스 관리
│   │   └── db_manager.py
│   ├── collectors/              # 데이터 수집기
│   │   └── evaluation_collector.py
│   └── utils/                   # 유틸리티
│       └── logger.py
├── data/                        # 데이터 저장소
│   └── llm_evaluations.db
├── reports/                     # 생성된 리포트
└── exports/                     # 내보낸 데이터
```

## ⚙️ 설정

### 환경 변수 (.env)

```bash
# HuggingFace API 토큰 (선택사항, 더 많은 API 호출 허용)
HUGGINGFACE_TOKEN=your_token_here

# 로그 레벨
LOG_LEVEL=INFO

# API 요청 간격 (초)
API_DELAY=0.1

# 태스크별 수집할 모델 수
MODELS_PER_TASK=30
```

### 설정 커스터마이징 (config.py)

```python
# 수집할 태스크 수정
TASKS_TO_COLLECT = [
    "text-generation",
    "text-classification", 
    "question-answering"
]

# 태스크별 모델 수 조정
MODELS_PER_TASK = 50

# API 요청 간격 조정
API_DELAY = 0.2
```

## 📈 사용 예시

### 1. 대화형 분석

```python
from src.collectors.evaluation_collector import LLMEvaluationCollector
import pandas as pd

collector = LLMEvaluationCollector()

# 텍스트 생성 모델 상위 10개 조회
models = collector.db_manager.get_top_models_by_downloads(
    limit=10, 
    task="text-generation"
)

print("상위 텍스트 생성 모델:")
print(models[['model_id', 'downloads', 'likes']].head())

# 특정 모델의 평가 결과 조회
evaluations = collector.db_manager.get_evaluations_by_model(
    "microsoft/DialoGPT-medium"
)

print("\n평가 결과:")
print(evaluations[['dataset_name', 'metric_name', 'metric_value']])
```

### 2. 배치 수집

```python
# 특정 태스크들만 수집
tasks_to_collect = ["text-generation", "question-answering"]

for task in tasks_to_collect:
    collector.collect_models_by_task(task, limit=20)
    
# 리포트 일괄 생성
for task in tasks_to_collect:
    collector.generate_task_report(task)
```

### 3. 커스텀 분석

```python
# 다운로드 수 vs 성능 분석
models_df = collector.db_manager.get_models_by_task("text-generation")
evaluations_df = collector.db_manager.get_evaluations_by_model("gpt2")

# 두 데이터프레임 병합하여 분석
merged_df = models_df.merge(
    evaluations_df, 
    left_on='model_id', 
    right_on='model_id'
)

# 다운로드 수와 성능의 상관관계 분석
correlation = merged_df['downloads'].corr(merged_df['metric_value'])
print(f"다운로드 수와 성능의 상관관계: {correlation:.3f}")
```

## 🔧 고급 사용법

### 1. 데이터베이스 직접 쿼리

```python
import sqlite3
import pandas as pd

# 데이터베이스 연결
conn = sqlite3.connect("data/llm_evaluations.db")

# 커스텀 쿼리
query = """
SELECT m.model_id, m.downloads, e.metric_value
FROM models m
JOIN evaluations e ON m.model_id = e.model_id
WHERE e.metric_name = 'accuracy' AND e.metric_value > 0.8
ORDER BY m.downloads DESC
"""

results = pd.read_sql_query(query, conn)
print(results)
```

### 2. 실시간 모니터링

```python
import time
from src.utils.logger import ProgressLogger, get_logger

logger = get_logger(__name__)

def monitor_collection():
    collector = LLMEvaluationCollector()
    
    while True:
        stats = collector.get_collection_summary()
        logger.info(f"현재 모델 수: {stats['database_stats']['total_models']}")
        
        # 1시간마다 새로운 모델 확인
        time.sleep(3600)
```

### 3. 데이터 검증

```python
from src.models.data_models import validate_model_info, validate_evaluation_result

# 데이터 품질 검사
def validate_database():
    collector = LLMEvaluationCollector()
    
    # 모든 모델 정보 검증
    models_df = collector.db_manager.get_top_models_by_downloads(limit=1000)
    
    invalid_models = []
    for _, row in models_df.iterrows():
        model_info = ModelInfo(**row.to_dict())
        if not validate_model_info(model_info):
            invalid_models.append(model_info.model_id)
    
    print(f"유효하지 않은 모델: {len(invalid_models)}개")
    return invalid_models
```

## 🐛 문제 해결

### 자주 발생하는 문제

1. **API Rate Limit 오류**
   ```bash
   # API 요청 간격을 늘려주세요
   export API_DELAY=0.5
   ```

2. **데이터베이스 락 오류**
   ```python
   # 데이터베이스 백업 후 재생성
   collector.db_manager.backup_database()
   ```

3. **메모리 부족**
   ```bash
   # 배치 크기를 줄여주세요
   python main.py --task text-generation --limit 10
   ```

### 로그 확인

```bash
# 상세 로깅 활성화
python main.py --collect-all --verbose

# 로그 파일 확인
tail -f logs/hf_llm_evaluation_*.log
```

**Happy Evaluating! 🚀**