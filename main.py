"""
HuggingFace LLM 평가 데이터 수집 프로젝트 메인 스크립트

사용법:
    python main.py --stats                     # 현재 수집 현황 확인
    python main.py --collect-all               # 모든 태스크 데이터 수집
    python main.py --task text-generation      # 특정 태스크만 수집
    python main.py --reports                   # 리포트 생성
    python main.py --export                    # 데이터 내보내기
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import json

# 프로젝트 모듈 import
from src.collectors.evaluation_collector import LLMEvaluationCollector
from src.utils.logger import init_project_logging, get_logger
from config import Config


def setup_environment():
    """환경 설정 및 초기화"""
    # 설정 검증 및 디렉토리 생성
    Config.validate()

    # 로깅 초기화
    logger = init_project_logging(Config.LOG_LEVEL)
    logger.info("=== HuggingFace LLM 평가 데이터 수집 프로젝트 시작 ===")
    logger.info(f"로그 레벨: {Config.LOG_LEVEL}")
    logger.info(f"데이터베이스 경로: {Config.DATABASE_PATH}")

    return logger


def collect_single_task(task: str, limit: int = None) -> bool:
    """단일 태스크의 모델 데이터 수집 (README 예시 구현)"""
    logger = get_logger(__name__)

    if task not in Config.TASKS_TO_COLLECT:
        logger.error(f"지원하지 않는 태스크: {task}")
        logger.info(f"지원 태스크: {', '.join(Config.TASKS_TO_COLLECT)}")
        return False

    try:
        collector = LLMEvaluationCollector()

        logger.info(f"태스크 '{task}' 데이터 수집 시작 (최대 {limit or Config.MODELS_PER_TASK}개 모델)")

        # 모델 수집
        models = collector.collect_models_by_task(task, limit or Config.MODELS_PER_TASK)

        if not models:
            logger.warning(f"태스크 '{task}'에서 수집된 모델이 없습니다.")
            return False

        logger.info(f"수집 완료: {len(models)}개 모델")

        # 상위 몇 개 모델의 평가 결과 수집
        top_models = models[:min(5, len(models))]
        evaluation_count = 0

        for model in top_models:
            evaluations = collector.collect_evaluations_for_model(model.model_id)
            evaluation_count += len(evaluations)

        logger.info(f"평가 결과 수집 완료: {evaluation_count}개")

        # 태스크 리포트 생성
        report_path = collector.generate_task_report(task)
        if report_path:
            logger.info(f"리포트 생성: {report_path}")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"태스크 '{task}' 수집 중 오류 발생: {e}", exc_info=True)
        return False


def collect_all_tasks(limit_per_task: int = None) -> bool:
    """모든 태스크의 데이터 수집 (README 예시 구현)"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        logger.info(f"전체 태스크 수집 시작: {Config.TASKS_TO_COLLECT}")

        # 전체 수집 실행
        stats = collector.collect_all_tasks(
            models_per_task=limit_per_task or Config.MODELS_PER_TASK
        )

        # 결과 출력 (README 형식)
        logger.info("\n" + "="*50)
        logger.info("수집 완료 요약")
        logger.info("="*50)
        logger.info(f"총 모델 수: {stats.total_models:,}개")
        logger.info(f"총 평가 결과: {stats.total_evaluations:,}개")
        logger.info(f"수집된 태스크: {', '.join(stats.tasks_collected)}")
        logger.info(f"오류 수: {stats.errors_count}개")
        logger.info(f"성공률: {stats.success_rate:.2%}")

        if stats.collection_duration:
            logger.info(f"소요 시간: {stats.collection_duration:.2f}초")

        # 자동으로 데이터 내보내기
        collector.export_data()
        logger.info("데이터 내보내기 완료")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"전체 수집 중 오류 발생: {e}", exc_info=True)
        return False


def generate_reports(tasks: Optional[List[str]] = None) -> bool:
    """태스크별 리포트 생성 (README 예시 구현)"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        if tasks is None:
            tasks = Config.TASKS_TO_COLLECT

        logger.info(f"리포트 생성 시작: {tasks}")

        success_count = 0
        for task in tasks:
            try:
                # 태스크별 상세 리포트
                report_path = collector.generate_task_report(task)
                if report_path:
                    logger.info(f"✓ {task} 리포트: {report_path}")
                    success_count += 1

                # 주요 메트릭별 리더보드 생성
                task_info = Config.get_task_info(task)
                main_metrics = task_info.get('common_metrics', [])

                for metric in main_metrics[:1]:  # 주요 메트릭 1개만
                    leaderboard_path = collector.generate_leaderboard(task, metric)
                    if leaderboard_path:
                        logger.info(f"✓ {task}-{metric} 리더보드: {leaderboard_path}")

            except Exception as e:
                logger.error(f"태스크 '{task}' 리포트 생성 실패: {e}")
                continue

        logger.info(f"리포트 생성 완료: {success_count}/{len(tasks)}개 태스크")
        collector.close()
        return success_count > 0

    except Exception as e:
        logger.error(f"리포트 생성 중 오류 발생: {e}")
        return False


def show_statistics() -> bool:
    """수집 현황 통계 표시 (README 예시 구현)"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()
        summary = collector.get_collection_summary()

        # README 스타일 출력
        print("\n" + "="*60)
        print("HuggingFace LLM 평가 데이터 수집 현황")
        print("="*60)

        db_stats = summary.get('database_stats', {})
        print(f"총 모델 수: {db_stats.get('total_models', 0):,}개")
        print(f"총 평가 결과: {db_stats.get('total_evaluations', 0):,}개")
        print(f"마지막 수집: {db_stats.get('last_collection', 'N/A')}")

        print(f"\n태스크별 모델 수:")
        task_counts = db_stats.get('task_counts', {})
        if task_counts:
            for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {task}: {count:,}개")
        else:
            print("  (데이터 없음)")

        print(f"\n지원하는 태스크:")
        for task in Config.TASKS_TO_COLLECT:
            task_info = Config.get_task_info(task)
            print(f"  • {task}: {task_info.get('description', '')}")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"통계 조회 중 오류 발생: {e}")
        return False


def export_data_only() -> bool:
    """데이터를 CSV로 내보내기 (README 예시 구현)"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        logger.info("데이터 내보내기 시작...")
        collector.export_data()

        # 내보낸 파일 목록 표시
        export_dir = Path(Config.EXPORTS_DIR)
        if export_dir.exists():
            csv_files = list(export_dir.glob("*.csv"))
            logger.info(f"내보내기 완료 ({len(csv_files)}개 파일):")
            for file_path in csv_files:
                logger.info(f"  • {file_path}")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"데이터 내보내기 중 오류 발생: {e}")
        return False


def main():
    """메인 함수 - README의 명령어 예시와 일치"""
    parser = argparse.ArgumentParser(
        description="HuggingFace LLM 성능 평가 데이터 수집 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시 (README와 일치):
  python main.py --stats                    # 현재 수집 현황 확인
  python main.py --collect-all              # 모든 태스크 데이터 수집
  python main.py --task text-generation     # 특정 태스크만 수집
  python main.py --reports                  # 리포트 생성
  python main.py --export                   # 데이터 내보내기

지원 태스크:
  text-generation, text-classification, question-answering, 
  summarization, translation
        """
    )

    # 명령어 그룹 (README 예시와 정확히 일치)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stats", action="store_true",
                       help="현재 수집 현황 확인")
    group.add_argument("--collect-all", action="store_true",
                       help="모든 태스크 데이터 수집")
    group.add_argument("--task", type=str, choices=Config.TASKS_TO_COLLECT,
                       help="특정 태스크만 수집")
    group.add_argument("--reports", action="store_true",
                       help="리포트 생성")
    group.add_argument("--export", action="store_true",
                       help="데이터 내보내기")

    # 추가 옵션
    parser.add_argument("--limit", type=int,
                        help=f"수집할 모델 수 (기본값: {Config.MODELS_PER_TASK})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 로깅")

    args = parser.parse_args()

    # 환경 설정
    if args.verbose:
        Config.LOG_LEVEL = "DEBUG"

    logger = setup_environment()

    try:
        success = False

        # README 예시와 일치하는 명령어 처리
        if args.stats:
            success = show_statistics()
        elif args.collect_all:
            success = collect_all_tasks(args.limit)
        elif args.task:
            success = collect_single_task(args.task, args.limit)
        elif args.reports:
            success = generate_reports()
        elif args.export:
            success = export_data_only()

        if success:
            logger.info("✅ 작업이 성공적으로 완료되었습니다!")
            return 0
        else:
            logger.error("❌ 작업 실행 중 오류가 발생했습니다.")
            return 1

    except KeyboardInterrupt:
        logger.warning("⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"❌ 예기치 않은 오류: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())