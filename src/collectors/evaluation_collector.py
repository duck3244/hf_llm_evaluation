"""
LLM 평가 데이터 수집기 모듈
README 예시와 일치하도록 수정됨
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
from ..utils.logger import get_logger, ProgressLogger
from config import Config

logger = get_logger(__name__)


class CollectionError(Exception):
    """데이터 수집 관련 예외"""
    pass


class LLMEvaluationCollector:
    """LLM 평가 데이터 수집기 메인 클래스 (README 예시 구현)"""

    def __init__(self, hf_token: Optional[str] = None, db_path: Optional[str] = None):
        """
        수집기 초기화

        Args:
            hf_token: HuggingFace API 토큰 (선택사항)
            db_path: 데이터베이스 파일 경로 (선택사항)
        """
        self.hf_api = get_api_client(hf_token or Config.HUGGINGFACE_TOKEN)
        self.db_manager = DatabaseManager(db_path or Config.DATABASE_PATH)
        self.config = Config()

        # 태스크 카테고리 초기화
        self._initialize_task_categories()

        # 수집 통계
        self.collection_stats = CollectionStats()

        logger.info("✅ LLM 평가 데이터 수집기 초기화 완료")

    def _initialize_task_categories(self):
        """태스크 카테고리를 데이터베이스에 초기화"""
        logger.debug("태스크 카테고리 초기화 중...")

        for task_name, task_info in self.config.TASK_CATEGORIES.items():
            task_category = TaskCategory(
                task_name=task_name,
                description=task_info["description"],
                common_datasets=task_info["common_datasets"],
                common_metrics=task_info["common_metrics"]
            )
            self.db_manager.insert_task_category(task_category)

        logger.debug("태스크 카테고리 초기화 완료")

    def collect_models_by_task(self, task: str, limit: int = None) -> List[ModelInfo]:
        """
        특정 태스크의 모델들을 수집 (README 예시 구현)

        Args:
            task: 태스크 이름 (예: "text-generation")
            limit: 수집할 모델 수 제한

        Returns:
            수집된 ModelInfo 객체 리스트
        """
        if limit is None:
            limit = self.config.MODELS_PER_TASK

        if task not in self.config.TASKS_TO_COLLECT:
            raise CollectionError(f"지원하지 않는 태스크: {task}")

        logger.info(f"📥 태스크 '{task}' 모델 수집 시작 (최대 {limit}개)")
        start_time = time.time()

        try:
            # HuggingFace API에서 모델 목록 가져오기
            models_data = self.hf_api.get_models(task=task, limit=limit)

            if not models_data:
                logger.warning(f"태스크 '{task}'에서 모델을 찾을 수 없습니다.")
                return []

            models = []
            errors = 0

            # 진행률 표시
            progress = ProgressLogger(len(models_data), logger, f"태스크 {task} 수집")

            for i, model_data in enumerate(models_data, 1):
                try:
                    model_id = model_data.get('id', 'unknown')
                    logger.debug(f"모델 처리 중: {model_id}")

                    # 상세 정보 가져오기
                    detailed_info = self.hf_api.get_model_info(model_id)
                    if not detailed_info:
                        logger.warning(f"모델 상세 정보를 가져올 수 없음: {model_id}")
                        errors += 1
                        progress.update()
                        continue

                    # ModelInfo 객체 생성
                    model_info = self._create_model_info(detailed_info, task)

                    # 데이터베이스에 저장
                    if self.db_manager.insert_model(model_info):
                        models.append(model_info)
                        logger.debug(f"✓ 모델 수집 완료: {model_info.model_id}")
                    else:
                        logger.warning(f"모델 저장 실패: {model_id}")
                        errors += 1

                    # API 제한 방지를 위한 지연
                    time.sleep(self.config.API_DELAY)
                    progress.update()

                except Exception as e:
                    logger.error(f"모델 처리 중 오류 ({model_data.get('id', 'unknown')}): {e}")
                    errors += 1
                    progress.update()
                    continue

            # 수집 완료
            progress.finish()
            duration = time.time() - start_time

            # 통계 업데이트
            self.collection_stats.total_models += len(models)
            self.collection_stats.errors_count += errors
            if task not in self.collection_stats.tasks_collected:
                self.collection_stats.tasks_collected.append(task)

            logger.info(f"✅ 태스크 '{task}' 수집 완료:")
            logger.info(f"   • 성공: {len(models)}개 모델")
            logger.info(f"   • 오류: {errors}개")
            logger.info(f"   • 소요시간: {duration:.2f}초")

            return models

        except Exception as e:
            logger.error(f"태스크 '{task}' 수집 중 치명적 오류: {e}")
            raise CollectionError(f"태스크 수집 실패: {e}")

    def collect_evaluations_for_model(self, model_id: str, max_evaluations: int = None) -> List[EvaluationResult]:
        """
        특정 모델의 평가 결과를 수집

        Args:
            model_id: 모델 ID
            max_evaluations: 최대 평가 결과 수

        Returns:
            수집된 EvaluationResult 객체 리스트
        """
        if max_evaluations is None:
            max_evaluations = self.config.MAX_EVALUATIONS_PER_MODEL

        logger.debug(f"📊 모델 '{model_id}' 평가 결과 수집 중...")

        try:
            evaluations = self.hf_api.get_model_evaluations(model_id)
            saved_evaluations = []

            for evaluation in evaluations[:max_evaluations]:
                if self.db_manager.insert_evaluation(evaluation):
                    saved_evaluations.append(evaluation)
                    logger.debug(f"✓ 평가 결과 저장: {evaluation.metric_name}={evaluation.metric_value}")

            self.collection_stats.total_evaluations += len(saved_evaluations)

            if saved_evaluations:
                logger.debug(f"✅ 모델 '{model_id}' 평가 결과 수집 완료: {len(saved_evaluations)}개")

            return saved_evaluations

        except Exception as e:
            logger.error(f"모델 '{model_id}' 평가 결과 수집 실패: {e}")
            return []

    def collect_all_tasks(self, tasks: Optional[List[str]] = None,
                          models_per_task: Optional[int] = None) -> CollectionStats:
        """
        모든 태스크의 데이터를 수집 (README 예시 구현)

        Args:
            tasks: 수집할 태스크 리스트 (None시 기본 태스크)
            models_per_task: 태스크당 수집할 모델 수

        Returns:
            수집 통계 정보
        """
        if tasks is None:
            tasks = self.config.TASKS_TO_COLLECT
        if models_per_task is None:
            models_per_task = self.config.MODELS_PER_TASK

        logger.info(f"🚀 전체 태스크 데이터 수집 시작")
        logger.info(f"   • 대상 태스크: {tasks}")
        logger.info(f"   • 태스크당 모델 수: {models_per_task}")

        start_time = time.time()
        self.collection_stats = CollectionStats()

        for task_idx, task in enumerate(tasks, 1):
            try:
                logger.info(f"\n📋 [{task_idx}/{len(tasks)}] 태스크 수집: {task}")

                # 모델 수집
                models = self.collect_models_by_task(task, models_per_task)

                if not models:
                    logger.warning(f"태스크 '{task}'에서 수집된 모델이 없습니다.")
                    continue

                # 상위 모델들의 평가 결과 수집
                top_models = models[:5]  # 상위 5개 모델만
                logger.info(f"📊 상위 {len(top_models)}개 모델의 평가 결과 수집 중...")

                for model in top_models:
                    self.collect_evaluations_for_model(model.model_id)
                    time.sleep(self.config.API_DELAY)

                logger.info(f"✅ 태스크 '{task}' 완료")

            except Exception as e:
                logger.error(f"❌ 태스크 '{task}' 수집 중 오류: {e}")
                self.collection_stats.errors_count += 1
                continue

        # 최종 통계 계산
        duration = time.time() - start_time
        self.collection_stats.collection_duration = duration
        self.collection_stats.update_success_rate()

        logger.info(f"\n🎉 전체 수집 완료!")
        logger.info(f"   • 총 모델: {self.collection_stats.total_models:,}개")
        logger.info(f"   • 총 평가 결과: {self.collection_stats.total_evaluations:,}개")
        logger.info(f"   • 수집 태스크: {', '.join(self.collection_stats.tasks_collected)}")
        logger.info(f"   • 소요 시간: {duration:.2f}초")
        logger.info(f"   • 성공률: {self.collection_stats.success_rate:.2%}")

        return self.collection_stats

    def generate_task_report(self, task: str, output_dir: Optional[str] = None) -> str:
        """
        태스크별 상세 리포트 생성 (README 형식)

        Args:
            task: 태스크 이름
            output_dir: 출력 디렉토리

        Returns:
            생성된 리포트 파일 경로
        """
        if output_dir is None:
            output_dir = self.config.REPORTS_DIR

        logger.info(f"📄 태스크 '{task}' 리포트 생성 중...")

        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # 데이터 수집
            models_df = self.db_manager.get_models_by_task(task)

            if models_df.empty:
                logger.warning(f"태스크 '{task}'에 대한 모델 데이터가 없습니다.")
                return ""

            task_info = self.config.get_task_info(task)

            # 리포트 내용 생성
            report_content = self._generate_report_content(task, models_df, task_info)

            # 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"{task}_report_{timestamp}.md"
            report_path = output_path / report_filename

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"✅ 리포트 생성 완료: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"리포트 생성 실패: {e}")
            return ""

    def generate_leaderboard(self, task_type: str, metric_name: str,
                             dataset_name: Optional[str] = None,
                             output_dir: Optional[str] = None) -> str:
        """
        태스크별 리더보드 생성 (README 형식)
        """
        if output_dir is None:
            output_dir = self.config.REPORTS_DIR

        logger.info(f"🏆 리더보드 생성: {task_type} - {metric_name}")

        try:
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

            logger.info(f"✅ 리더보드 저장 완료: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"리더보드 생성 실패: {e}")
            return ""

    def export_data(self, output_dir: Optional[str] = None):
        """수집된 데이터를 CSV로 내보내기 (README 예시 구현)"""
        if output_dir is None:
            output_dir = self.config.EXPORTS_DIR

        logger.info(f"📁 데이터 내보내기 시작: {output_dir}")

        try:
            self.db_manager.export_to_csv(output_dir)
            logger.info(f"✅ 데이터 내보내기 완료: {output_dir}")
        except Exception as e:
            logger.error(f"데이터 내보내기 실패: {e}")

    def get_collection_summary(self) -> Dict[str, Any]:
        """수집 현황 요약 (README 예시 구현)"""
        try:
            stats = self.db_manager.get_model_statistics()

            summary = {
                "database_stats": stats,
                "collection_stats": self.collection_stats.to_dict(),
                "supported_tasks": list(self.config.TASK_CATEGORIES.keys()),
                "last_updated": datetime.now().isoformat()
            }

            return summary
        except Exception as e:
            logger.error(f"수집 요약 생성 실패: {e}")
            return {}

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
        """태스크 리포트 내용 생성 (README 스타일)"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        content = f"""# {task.upper()} 태스크 성능 분석 리포트

**생성일:** {timestamp}

## 📋 태스크 개요

**설명:** {task_info.get('description', '설명 없음')}

**주요 데이터셋:** {', '.join(task_info.get('common_datasets', []))}

**주요 메트릭:** {', '.join(task_info.get('common_metrics', []))}

## 📊 수집 통계

- **총 모델 수:** {len(models_df):,}개
- **평균 다운로드 수:** {models_df['downloads'].mean():,.0f}
- **평균 좋아요 수:** {models_df['likes'].mean():,.1f}

## 🏆 상위 10개 모델 (다운로드 순)

| 순위 | 모델 ID | 작성자 | 다운로드 | 좋아요 | 크기 |
|------|---------|--------|----------|--------|------|
"""

        top_models = models_df.head(10)
        for idx, (_, row) in enumerate(top_models.iterrows(), 1):
            content += f"| {idx} | {row['model_id']} | {row['author']} | {row['downloads']:,} | {row['likes']:,} | {row.get('model_size', 'N/A')} |\n"

        # 작성자별 통계
        author_stats = models_df['author'].value_counts().head(5)
        content += f"\n## 👥 상위 작성자\n\n"
        for author, count in author_stats.items():
            content += f"- **{author}:** {count}개 모델\n"

        # 라이브러리별 분포
        if 'library_name' in models_df.columns:
            library_stats = models_df['library_name'].value_counts().head(5)
            content += f"\n## 🔧 주요 라이브러리\n\n"
            for library, count in library_stats.items():
                if library:
                    content += f"- **{library}:** {count}개 모델\n"

        content += f"\n---\n*이 리포트는 HuggingFace LLM 평가 데이터 수집 도구로 생성되었습니다.*"

        return content

    def _generate_leaderboard_content(self, task_type: str, metric_name: str,
                                      dataset_name: Optional[str], leaderboard_df: pd.DataFrame) -> str:
        """리더보드 내용 생성 (README 스타일)"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        title = f"{task_type.upper()} - {metric_name.upper()}"
        if dataset_name:
            title += f" ({dataset_name})"

        content = f"""# 🏆 {title} 리더보드

**생성일:** {timestamp}

**태스크:** {task_type}
**메트릭:** {metric_name}
"""

        if dataset_name:
            content += f"**데이터셋:** {dataset_name}\n"

        content += f"\n**총 모델 수:** {len(leaderboard_df)}개\n\n"

        # 리더보드 테이블
        content += "## 📈 순위\n\n"
        content += "| 순위 | 모델 | 작성자 | 점수 | 다운로드 | 검증됨 |\n"
        content += "|------|------|--------|------|----------|--------|\n"

        for idx, (_, row) in enumerate(leaderboard_df.iterrows(), 1):
            verified = "✅" if row.get('verified', False) else "❌"
            content += f"| {idx} | {row['model_id']} | {row['author']} | {row['metric_value']:.4f} | {row['downloads']:,} | {verified} |\n"

        content += f"\n---\n*이 리더보드는 HuggingFace LLM 평가 데이터 수집 도구로 생성되었습니다.*"

        return content

    def close(self):
        """리소스 정리"""
        logger.info("🔒 수집기 리소스 정리 중...")
        try:
            self.db_manager.close()
            logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            logger.error(f"리소스 정리 중 오류: {e}")


# README의 편의 함수들 구현
def quick_collect_task(task: str, limit: int = 20, hf_token: Optional[str] = None) -> List[ModelInfo]:
    """
    특정 태스크의 모델을 빠르게 수집 (README 예시 구현)

    Args:
        task: 태스크 이름
        limit: 수집할 모델 수
        hf_token: HuggingFace API 토큰

    Returns:
        수집된 모델 리스트
    """
    collector = LLMEvaluationCollector(hf_token=hf_token)
    try:
        logger.info(f"🚀 빠른 수집: {task} ({limit}개 모델)")
        return collector.collect_models_by_task(task, limit)
    finally:
        collector.close()


def generate_task_summary(task: str) -> Dict[str, Any]:
    """
    태스크 요약 정보 생성 (README 예시 구현)

    Args:
        task: 태스크 이름

    Returns:
        태스크 요약 정보
    """
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
    """
    여러 모델 비교 (README 예시 구현)

    Args:
        model_ids: 비교할 모델 ID 리스트

    Returns:
        모델 비교 데이터프레임
    """
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