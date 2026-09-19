from pathlib import Path
from datetime import datetime
import csv
import time

from predict_cmp import predict_cmp_folder


PROJECT_DIR = Path(__file__).resolve().parent
INCOMING_DIR = PROJECT_DIR / "incoming_raw"
PREDICTION_DIR = PROJECT_DIR / "predictions"
LOG_DIR = PROJECT_DIR / "logs"
LOG_PATH = LOG_DIR / "pipeline_history.csv"

# CSV가 10초 동안 변하지 않으면 저장 완료로 판단
STABLE_SECONDS = 10

# 배치별 파일 상태 저장
batch_states = {}

# 필요한 폴더 자동 생성
INCOMING_DIR.mkdir(exist_ok=True)
PREDICTION_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def get_csv_snapshot(batch_folder):
    """CSV 파일명·크기·수정시간을 확인한다."""

    csv_files = sorted(batch_folder.glob("*.csv"))

    snapshot = tuple(
        (
            file.name,
            file.stat().st_size,
            file.stat().st_mtime_ns
        )
        for file in csv_files
    )

    return csv_files, snapshot


def write_pipeline_log(
    batch_name,
    csv_count,
    wafer_count,
    status,
    output_path="",
    error_message=""
):
    """자동 처리 결과를 pipeline_history.csv에 기록한다."""

    log_exists = LOG_PATH.exists()

    with LOG_PATH.open(
        mode="a",
        newline="",
        encoding="utf-8-sig"
    ) as log_file:

        writer = csv.writer(log_file)

        # 처음 생성할 때만 제목 행 작성
        if not log_exists:
            writer.writerow([
                "처리시간",
                "배치명",
                "CSV개수",
                "예측웨이퍼수",
                "처리상태",
                "결과파일",
                "오류내용"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            batch_name,
            csv_count,
            wafer_count,
            status,
            str(output_path),
            str(error_message)
        ])


print("CMP Raw Data 자동 감시 시작")
print("감시 폴더:", INCOMING_DIR)
print(f"CSV가 {STABLE_SECONDS}초간 변하지 않으면 자동 실행합니다.")
print("실행 기록:", LOG_PATH)
print("종료하려면 Ctrl + C를 누르세요.")


while True:
    current_time_seconds = time.time()

    for batch_folder in INCOMING_DIR.iterdir():

        if not batch_folder.is_dir():
            continue

        completed_file = batch_folder / "COMPLETED.txt"

        # 이미 처리된 배치는 건너뜀
        if completed_file.exists():
            continue

        csv_files, current_snapshot = get_csv_snapshot(
            batch_folder
        )

        if not csv_files:
            continue

        previous_state = batch_states.get(
            batch_folder.name
        )

        # 처음 발견했거나 파일 상태가 달라진 경우
        if (
            previous_state is None
            or previous_state["snapshot"] != current_snapshot
        ):
            batch_states[batch_folder.name] = {
                "snapshot": current_snapshot,
                "last_changed": current_time_seconds
            }

            print(
                f"CSV 변화 감지: {batch_folder.name} "
                f"({len(csv_files)}개)"
            )

            continue

        stable_time = (
            current_time_seconds
            - previous_state["last_changed"]
        )

        if stable_time < STABLE_SECONDS:
            continue

        print()
        print("CSV 저장 완료 판단:", batch_folder.name)
        print("자동 예측 시작")

        output_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_path = (
            PREDICTION_DIR
            / f"{batch_folder.name}_{output_time}_prediction.csv"
        )

        try:
            # 자동 전처리 및 예측
            prediction_result = predict_cmp_folder(
                raw_folder=batch_folder,
                output_path=output_path
            )

            wafer_count = len(prediction_result)

            # 성공 이력 저장
            write_pipeline_log(
                batch_name=batch_folder.name,
                csv_count=len(csv_files),
                wafer_count=wafer_count,
                status="SUCCESS",
                output_path=output_path
            )

            # 중복 처리 방지 파일 생성
            completed_file.write_text(
                f"처리 완료: {output_time}",
                encoding="utf-8"
            )

            print("배치 처리 완료:", batch_folder.name)
            print("실행 이력 저장 완료")

            batch_states.pop(
                batch_folder.name,
                None
            )

        except Exception as error:
            print("배치 처리 실패:", batch_folder.name)
            print("오류 내용:", error)

            # 실패 이력도 저장
            write_pipeline_log(
                batch_name=batch_folder.name,
                csv_count=len(csv_files),
                wafer_count=0,
                status="FAILED",
                output_path=output_path,
                error_message=error
            )

            # 10초 후 재시도
            batch_states[batch_folder.name][
                "last_changed"
            ] = time.time()

    time.sleep(2)