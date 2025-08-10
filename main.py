"""
HuggingFace LLM 평가 데이터 수집 프로젝트 메인 스크립트
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
    """환경 설정"""
    # 설정 검증
    Config.validate()

    # 로깅 초기화
    logger = init_project_logging(Config.LOG_LEVEL)
    logger.info("HuggingFace LLM 평가 데이터 수집 프로젝트 시작")

    return logger


def collect_single_task(task: str, limit: int = None) -> bool:
    """단일 태스크 수집"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        logger.info(f"태스크 '{task}' 수집 시작")
        models = collector.collect_models_by_task(task, limit or Config.MODELS_PER_TASK)

        # 상위 모델들의 평가 결과 수집
        top_models = models[:5]
        for model in top_models:
            collector.collect_evaluations_for_model(model.model_id)

        # 리포트 생성
        report_path = collector.generate_task_report(task)
        logger.info(f"리포트 생성 완료: {report_path}")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"태스크 '{task}' 수집 실패: {e}")
        return False


def collect_all_tasks(limit_per_task: int = None) -> bool:
    """모든 태스크 수집"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        # 전체 수집 실행
        stats = collector.collect_all_tasks(
            models_per_task=limit_per_task or Config.MODELS_PER_TASK
        )

        # 결과 출력
        logger.info("=== 수집 완료 통계 ===")
        logger.info(f"총 모델: {stats.total_models}개")
        logger.info(f"총 평가 결과: {stats.total_evaluations}개")
        logger.info(f"수집 태스크: {', '.join(stats.tasks_collected)}")
        logger.info(f"성공률: {stats.success_rate:.2%}")

        # 데이터 내보내기
        collector.export_data()

        collector.close()
        return True

    except Exception as e:
        logger.error(f"전체 수집 실패: {e}")
        return False


def update_existing_data() -> bool:
    """기존 데이터 업데이트"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        logger.info("기존 모델 데이터 업데이트 시작")
        updated_count = collector.update_model_data()

        logger.info(f"데이터 업데이트 완료: {updated_count}개 모델")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"데이터 업데이트 실패: {e}")
        return False


def generate_reports(tasks: Optional[List[str]] = None) -> bool:
    """리포트 생성"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        if tasks is None:
            tasks = Config.TASKS_TO_COLLECT

        for task in tasks:
            try:
                report_path = collector.generate_task_report(task)
                logger.info(f"리포트 생성: {report_path}")
            except Exception as e:
                logger.error(f"태스크 '{task}' 리포트 생성 실패: {e}")
                continue

        collector.close()
        return True

    except Exception as e:
        logger.error(f"리포트 생성 실패: {e}")
        return False


def generate_leaderboards() -> bool:
    """리더보드 생성"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()

        # 주요 태스크-메트릭 조합
        leaderboard_configs = [
            ("text-generation", "accuracy"),
            ("text-classification", "f1"),
            ("question-answering", "f1"),
            ("summarization", "rouge-1"),
            ("translation", "bleu")
        ]

        for task_type, metric in leaderboard_configs:
            try:
                leaderboard_path = collector.generate_leaderboard(task_type, metric)
                if leaderboard_path:
                    logger.info(f"리더보드 생성: {leaderboard_path}")
            except Exception as e:
                logger.error(f"리더보드 생성 실패 ({task_type}-{metric}): {e}")
                continue

        collector.close()
        return True

    except Exception as e:
        logger.error(f"리더보드 생성 실패: {e}")
        return False


def show_statistics() -> bool:
    """수집 통계 표시"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()
        summary = collector.get_collection_summary()

        print("\n=== HuggingFace LLM 평가 데이터 수집 현황 ===")
        print(f"총 모델 수: {summary['database_stats'].get('total_models', 0):,}개")
        print(f"총 평가 결과: {summary['database_stats'].get('total_evaluations', 0):,}개")
        print(f"마지막 수집: {summary['database_stats'].get('last_collection', 'N/A')}")

        print("\n태스크별 모델 수:")
        task_counts = summary['database_stats'].get('task_counts', {})
        for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {task}: {count:,}개")

        print(f"\n지원 태스크: {', '.join(summary['supported_tasks'])}")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        return False


def export_data_only() -> bool:
    """데이터만 내보내기"""
    logger = get_logger(__name__)

    try:
        collector = LLMEvaluationCollector()
        collector.export_data()
        logger.info("데이터 내보내기 완료")

        collector.close()
        return True

    except Exception as e:
        logger.error(f"데이터 내보내기 실패: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="HuggingFace LLM 성능 평가 데이터 수집 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --collect-all              # 모든 태스크 수집
  python main.py --task text-generation     # 특정 태스크만 수집
  python main.py --update                   # 기존 데이터 업데이트
  python main.py --reports                  # 리포트만 생성
  python main.py --stats                    # 통계 표시
  python main.py --export                   # 데이터 내보내기
        """
    )

    # 명령어 그룹
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-all", action="store_true",
                       help="모든 태스크의 데이터 수집")
    group.add_argument("--task", type=str,
                       help="특정 태스크만 수집 (예: text-generation)")
    group.add_argument("--update", action="store_true",
                       help="기존 모델 데이터 업데이트")
    group.add_argument("--reports", action="store_true",
                       help="리포트만 생성")
    group.add_argument("--leaderboards", action="store_true",
                       help="리더보드만 생성")
    group.add_argument("--stats", action="store_true",
                       help="현재 수집 통계 표시")
    group.add_argument("--export", action="store_true",
                       help="데이터를 CSV로 내보내기")

    # 옵션
    parser.add_argument("--limit", type=int,
                        help=f"태스크당 수집할 모델 수 (기본값: {Config.MODELS_PER_TASK})")
    parser.add_argument("--token", type=str,
                        help="HuggingFace API 토큰")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 로깅 활성화")

    args = parser.parse_args()

    # 환경 설정
    if args.verbose:
        Config.LOG_LEVEL = "DEBUG"

    if args.token:
        os.environ["HUGGINGFACE_TOKEN"] = args.token

    logger = setup_environment()

    try:
        success = False

        if args.collect_all:
            success = collect_all_tasks(args.limit)
        elif args.task:
            if args.task in Config.TASKS_TO_COLLECT:
                success = collect_single_task(args.task, args.limit)
            else:
                logger.error(f"지원하지 않는 태스크: {args.task}")
                logger.info(f"지원 태스크: {', '.join(Config.TASKS_TO_COLLECT)}")
                return 1
        elif args.update:
            success = update_existing_data()
        elif args.reports:
            success = generate_reports()
        elif args.leaderboards:
            success = generate_leaderboards()
        elif args.stats:
            success = show_statistics()
        elif args.export:
            success = export_data_only()

        if success:
            logger.info("작업이 성공적으로 완료되었습니다!")
            return 0
        else:
            logger.error("작업 실행 중 오류가 발생했습니다.")
            return 1

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"예기치 않은 오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())