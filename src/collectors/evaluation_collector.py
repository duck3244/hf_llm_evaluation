"""
LLM 평가 데이터 수집기 모듈
HuggingFace API와 데이터베이스를 연결하여 데이터 수집 및 저장
"""

import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd

from ..api.huggingface_api import HuggingFaceAPI, get_api_client
from ..database.db_manager import DatabaseManager
from ..models.data_models import (
    ModelInfo, EvaluationResult, TaskCategory, CollectionStats
)
from ..utils.logger import get_logger
from config import Config

logger = get_logger(__name__)


class CollectionError(Exception):
    """데이터 수집 관련 예외"""
    pass


class LLMEvaluationCollector:
    """LLM 평가 데이터 수집기 메인 클래스"""

    def __init__(self, hf_token: Optional[str] = None, db_path: Optional[str] = None):
        self.hf_api = get_api_client(hf_token or Config.HUGGINGFACE_TOKEN)
        self.db_manager = DatabaseManager(db_path or Config.DATABASE_PATH)
        self.config = Config()

        # 태스크 카테고리 초기화
        self._initialize_task_categories()

        # 수집 통계
        self.collection_stats = CollectionStats()

        logger.info("LLM 평가 데이터 수집기 초기화 완료")

    def _initialize_task_categories(self):
        """태스크 카테고리를 데이터베이스에 초기화"""
        logger.info("태스크 카테고리 초기화")

        for task_name, task_info in self.config.TASK_CATEGORIES.items():
            task_category = TaskCategory(
                task_name=task_name,
                description=task_info["description"],
                common_datasets=task_info["common_datasets"],
                common_metrics=task_info["common_metrics"]
            )
            self.db_manager.insert_task_category(task_category)

    def collect_models_by_task(self, task: str, limit: int = None) -> List[ModelInfo]:
        """특정 태스크의 모델들을 수집"""
        if limit is None:
            limit = self.config.MODELS_PER_TASK

        logger.info(f"태스크 '{task}' 모델 수집 시작 (최대 {limit}개)")
        start_time = time.time()

        models_data = self.hf_api.get_models(task=task, limit=limit)
        models = []
        errors = 0

        for i, model_data in enumerate(models_data, 1):
            try:
                logger.debug(f"모델 처리 중 ({i}/{len(models_data)}): {model_data.get('id', 'unknown')}")

                # 상세 정보 가져오기
                detailed_info = self.hf_api.get_model_info(model_data['id'])
                if not detailed_info:
                    errors += 1
                    continue

                # ModelInfo 객체 생성
                model_info = self._create_model_info(detailed_info, task)

                # 데이터베이스에 저장
                if self.db_manager.insert_model(model_info):
                    models.append(model_info)
                    logger.debug(f"모델 수집 완료: {model_info.model_id}")
                else:
                    errors += 1

                # API 제한 방지
                time.sleep(self.config.API_DELAY)

            except Exception as e:
                logger.error(f"모델 처리 중 오류 ({model_data.get('id', 'unknown')}): {e}")
                errors += 1
                continue

        # 통계 업데이트
        duration = time.time() - start_time
        self.collection_stats.total_models += len(models)
        self.collection_stats.errors_count += errors
        self.collection_stats.tasks_collected.append(task)

        logger.info(f"태스크 '{task}' 수집 완료: {len(models)}개 모델, {errors}개 오류, {duration:.2f}초")
        return models

    def collect_evaluations_for_model(self, model_id: str, max_evaluations: int = None) -> List[EvaluationResult]:
        """특정 모델의 평가 결과를 수집"""
        if max_evaluations is None:
            max_evaluations = self.config.MAX_EVALUATIONS_PER_MODEL

        logger.info(f"모델 '{model_id}' 평가 결과 수집")

        try:
            evaluations = self.hf_api.get_model_evaluations(model_id)
            saved_evaluations = []

            for evaluation in evaluations[:max_evaluations]:
                if self.db_manager.insert_evaluation(evaluation):
                    saved_evaluations.append(evaluation)
                    logger.debug(f"평가 결과 저장: {evaluation.metric_name}={evaluation.metric_value}")

            self.collection_stats.total_evaluations += len(saved_evaluations)
            logger.info(f"모델 '{model_id}' 평가 결과 수집 완료: {len(saved_evaluations)}개")

            return saved_evaluations

        except Exception as e:
            logger.error(f"평가 결과 수집 실패 ({model_id}): {e}")
            return []

    def collect_all_tasks(self, tasks: Optional[List[str]] = None,
                          models_per_task: Optional[int] = None) -> CollectionStats:
        """모든 태스크의 데이터를 수집"""
        if tasks is None:
            tasks = self.config.TASKS_TO_COLLECT
        if models_per_task is None:
            models_per_task = self.config.MODELS_PER_TASK

        logger.info(f"전체 태스크 수집 시작: {tasks}")
        start_time = time.time()

        self.collection_stats = CollectionStats()

        for task in tasks:
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"태스크 수집: {task}")
                logger.info(f"{'=' * 60}")

                # 모델 수집
                models = self.collect_models_by_task(task, models_per_task)

                # 상위 모델들의 평가 결과 수집
                top_models = models[:5]  # 상위 5개 모델만
                for model in top_models:
                    self.collect_evaluations_for_model(model.model_id)
                    time.sleep(self.config.API_DELAY)

            except Exception as e:
                logger.error(f"태스크 '{task}' 수집 중 오류: {e}")
                self.collection_stats.errors_count += 1
                continue

        # 최종 통계 계산
        duration = time.time() - start_time
        self.collection_stats.collection_duration = duration
        self.collection_stats.update_success_rate()

        logger.info(f"\n전체 수집 완료:")
        logger.info(f"- 총 모델: {self.collection_stats.total_models}개")
        logger.info(f"- 총 평가 결과: {self.collection_stats.total_evaluations}개")
        logger.info(f"- 수집 태스크: {', '.join(self.collection_stats.tasks_collected)}")
        logger.info(f"- 소요 시간: {duration:.2f}초")
        logger.info(f"- 성공률: {self.collection_stats.success_rate:.2%}")

        return self.collection_stats

    def update_model_data(self, model_ids: Optional[List[str]] = None,
                          batch_size: int = 10) -> int:
        """기존 모델 데이터 업데이트"""
        if model_ids is None:
            # 데이터베이스에서 모든 모델 ID 가져오기
            models_df = self.db_manager.get_top_models_by_downloads(limit=100)
            model_ids = models_df['model_id'].tolist()

        logger.info(f"{len(model_ids)}개 모델 데이터 업데이트 시작")

        updated_count = 0
        for i in range(0, len(model_ids), batch_size):
            batch = model_ids[i:i + batch_size]

            for model_id in batch:
                try:
                    model_data = self.hf_api.get_model_info(model_id)
                    if model_data:
                        model_info = self._create_model_info(model_data)
                        if self.db_manager.insert_model(model_info):
                            updated_count += 1
                            logger.debug(f"모델 업데이트 완료: {model_id}")

                    time.sleep(self.config.API_DELAY)

                except Exception as e:
                    logger.error(f"모델 업데이트 실패 ({model_id}): {e}")
                    continue

        logger.info(f"모델 데이터 업데이트 완료: {updated_count}개")
        return updated_count

    def generate_task_report(self, task: str, output_dir: Optional[str] = None) -> str:
        """태스크별 상세 리포트 생성"""
        if output_dir is None:
            output_dir = self.config.REPORTS_DIR

        logger.info(f"태스크 '{task}' 리포트 생성")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 데이터 수집
        models_df = self.db_manager.get_models_by_task(task)
        task_info = self.config.get_task_info(task)

        # 리포트 내용 생성
        report_content = self._generate_report_content(task, models_df, task_info)

        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"{task}_report_{timestamp}.md"
        report_path = output_path / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"리포트 저장 완료: {report_path}")
        return str(report_path)

    def generate_leaderboard(self, task_type: str, metric_name: str,
                             dataset_name: Optional[str] = None,
                             output_dir: Optional[str] = None) -> str:
        """태스크별 리더보드 생성"""
        if output_dir is None:
            output_dir = self.config.REPORTS_DIR

        logger.info(f"리더보드 생성: {task_type} - {metric_name}")

        # 리더보드 데이터 가져오기
        leaderboard_df = self.db_manager.get_task_leaderboard(
            task_type, metric_name, dataset_name
        )

        if leaderboard_df.empty:
            logger.warning(f"리더보드 데이터가 없습니다: {task_type} - {metric_name}")
            return ""

        # 리더보드 리포트 생성
        report_content = self._generate_leaderboard_content(
            task_type, metric_name, dataset_name, leaderboard_df
        )

        # 파일 저장
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"leaderboard_{task_type}_{metric_name}_{timestamp}.md"
        if dataset_name:
            filename = f"leaderboard_{task_type}_{metric_name}_{dataset_name}_{timestamp}.md"

        report_path = output_path / filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"리더보드 저장 완료: {report_path}")
        return str(report_path)

    def export_data(self, output_dir: Optional[str] = None):
        """수집된 데이터를 CSV로 내보내기"""
        if output_dir is None:
            output_dir = self.config.EXPORTS_DIR

        logger.info("데이터 내보내기 시작")
        self.db_manager.export_to_csv(output_dir)
        logger.info(f"데이터 내보내기 완료: {output_dir}")

    def get_collection_summary(self) -> Dict[str, Any]:
        """수집 현황 요약"""
        stats = self.db_manager.get_model_statistics()

    def get_collection_summary(self) -> Dict[str, Any]:
        """수집 현황 요약"""
        stats = self.db_manager.get_model_statistics()

        summary = {
            "database_stats": stats,
            "collection_stats": self.collection_stats.to_dict(),
            "supported_tasks": list(self.config.TASK_CATEGORIES.keys()),
            "last_updated": datetime.now().isoformat()
        }

        return summary

    def _create_model_info(self, model_data: Dict[str, Any], task: Optional[str] = None) -> ModelInfo:
        """API 응답을 ModelInfo 객체로 변환"""
        task_categories = []
        if task:
            task_categories.append(task)

        # 태그에서 추가 태스크 추출
        tags = model_data.get('tags', [])
        for tag in tags:
            if tag in self.config.TASK_CATEGORIES:
                task_categories.append(tag)

        return ModelInfo(
            model_id=model_data.get('id', ''),
            model_name=model_data.get('id', '').split('/')[-1],
            author=model_data.get('author', ''),
            downloads=model_data.get('downloads', 0),
            likes=model_data.get('likes', 0),
            created_at=model_data.get('createdAt', ''),
            last_modified=model_data.get('lastModified', ''),
            library_name=model_data.get('library_name', ''),
            pipeline_tag=model_data.get('pipeline_tag', ''),
            tags=tags,
            task_categories=list(set(task_categories)),  # 중복 제거
            model_size=self._extract_model_size(model_data),
            license=model_data.get('cardData', {}).get('license', '')
        )

    def _extract_model_size(self, model_data: Dict[str, Any]) -> Optional[str]:
        """모델 크기 정보 추출"""
        tags = model_data.get('tags', [])
        size_patterns = ['7b', '13b', '30b', '65b', '70b', '175b', 'small', 'base', 'large', 'xl']

        for tag in tags:
            tag_lower = tag.lower()
            for pattern in size_patterns:
                if pattern in tag_lower:
                    return tag

        # 모델 이름에서 크기 정보 추출
        model_name = model_data.get('id', '').lower()
        for pattern in size_patterns:
            if pattern in model_name:
                return pattern

        return None

    def _generate_report_content(self, task: str, models_df: pd.DataFrame,
                                 task_info: Dict[str, Any]) -> str:
        """태스크 리포트 내용 생성"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        content = f"""# {task.upper()} 태스크 성능 분석 리포트

**생성일:** {timestamp}

## 태스크 개요

**설명:** {task_info.get('description', '설명 없음')}

**주요 데이터셋:** {', '.join(task_info.get('common_datasets', []))}

**주요 메트릭:** {', '.join(task_info.get('common_metrics', []))}

## 수집 통계

- **총 모델 수:** {len(models_df):,}개
- **평균 다운로드 수:** {models_df['downloads'].mean():,.0f}
- **평균 좋아요 수:** {models_df['likes'].mean():,.1f}

## 상위 10개 모델 (다운로드 순)

| 순위 | 모델 ID | 작성자 | 다운로드 | 좋아요 | 크기 |
|------|---------|--------|----------|--------|------|
"""

        top_models = models_df.head(10)
        for idx, (_, row) in enumerate(top_models.iterrows(), 1):
            content += f"| {idx} | {row['model_id']} | {row['author']} | {row['downloads']:,} | {row['likes']:,} | {row.get('model_size', 'N/A')} |\n"

        # 작성자별 통계
        author_stats = models_df['author'].value_counts().head(5)
        content += f"\n## 상위 작성자\n\n"
        for author, count in author_stats.items():
            content += f"- **{author}:** {count}개 모델\n"

        # 라이브러리별 분포
        if 'library_name' in models_df.columns:
            library_stats = models_df['library_name'].value_counts().head(5)
            content += f"\n## 주요 라이브러리\n\n"
            for library, count in library_stats.items():
                if library:
                    content += f"- **{library}:** {count}개 모델\n"

        return content

    def _generate_leaderboard_content(self, task_type: str, metric_name: str,
                                      dataset_name: Optional[str], leaderboard_df: pd.DataFrame) -> str:
        """리더보드 내용 생성"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        title = f"{task_type.upper()} - {metric_name.upper()}"
        if dataset_name:
            title += f" ({dataset_name})"

        content = f"""# {title} 리더보드

**생성일:** {timestamp}

**태스크:** {task_type}
**메트릭:** {metric_name}
"""

        if dataset_name:
            content += f"**데이터셋:** {dataset_name}\n"

        content += f"\n**총 모델 수:** {len(leaderboard_df)}개\n\n"

        # 리더보드 테이블
        content += "## 순위\n\n"
        content += "| 순위 | 모델 | 작성자 | 점수 | 다운로드 | 검증됨 |\n"
        content += "|------|------|--------|------|----------|--------|\n"

        for idx, (_, row) in enumerate(leaderboard_df.iterrows(), 1):
            verified = "✅" if row.get('verified', False) else "❌"
            content += f"| {idx} | {row['model_id']} | {row['author']} | {row['metric_value']:.4f} | {row['downloads']:,} | {verified} |\n"

        return content

    def close(self):
        """리소스 정리"""
        logger.info("수집기 리소스 정리")
        self.db_manager.close()


# 편의 함수들
def quick_collect_task(task: str, limit: int = 20, hf_token: Optional[str] = None) -> List[ModelInfo]:
    """특정 태스크의 모델을 빠르게 수집"""
    collector = LLMEvaluationCollector(hf_token=hf_token)
    try:
        return collector.collect_models_by_task(task, limit)
    finally:
        collector.close()


def generate_task_summary(task: str) -> Dict[str, Any]:
    """태스크 요약 정보 생성"""
    collector = LLMEvaluationCollector()
    try:
        models_df = collector.db_manager.get_models_by_task(task)

        if models_df.empty:
            return {"error": f"태스크 '{task}'에 대한 데이터가 없습니다."}

        return {
            "task": task,
            "total_models": len(models_df),
            "top_model": models_df.iloc[0]['model_id'] if not models_df.empty else None,
            "avg_downloads": models_df['downloads'].mean(),
            "total_downloads": models_df['downloads'].sum(),
            "unique_authors": models_df['author'].nunique()
        }
    finally:
        collector.close()


def get_model_comparison(model_ids: List[str]) -> pd.DataFrame:
    """여러 모델 비교"""
    collector = LLMEvaluationCollector()
    try:
        comparison_data = []

        for model_id in model_ids:
            evaluations_df = collector.db_manager.get_evaluations_by_model(model_id)

            if not evaluations_df.empty:
                # 각 메트릭의 최고 점수
                for _, eval_row in evaluations_df.iterrows():
                    comparison_data.append({
                        'model_id': model_id,
                        'dataset': eval_row['dataset_name'],
                        'metric': eval_row['metric_name'],
                        'value': eval_row['metric_value'],
                        'task_type': eval_row['task_type']
                    })

        return pd.DataFrame(comparison_data)
    finally:
        collector.close()