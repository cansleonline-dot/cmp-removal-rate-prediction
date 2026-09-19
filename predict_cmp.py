from pathlib import Path
import sys
import joblib
import pandas as pd


# 이 파일이 위치한 CMP_Project 폴더
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "models"
    / "cmp_removal_rate_model.joblib"
)

FEATURE_PATH = (
    PROJECT_DIR
    / "results"
    / "models"
    / "cmp_feature_columns.joblib"
)


process_columns = [
    "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE",
    "SLURRY_FLOW_LINE_A",
    "SLURRY_FLOW_LINE_B",
    "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION",
    "STAGE_ROTATION",
    "HEAD_ROTATION"
]


wear_columns = [
    "USAGE_OF_BACKING_FILM",
    "USAGE_OF_DRESSER",
    "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_DRESSER_TABLE",
    "USAGE_OF_MEMBRANE",
    "USAGE_OF_PRESSURIZED_SHEET"
]


all_columns = (
    ["WAFER_ID", "STAGE", "CHAMBER"]
    + process_columns
    + wear_columns
)


def predict_cmp_folder(raw_folder, output_path):
    raw_folder = Path(raw_folder)
    output_path = Path(output_path)

    # 저장된 모델 불러오기
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    # 폴더 안의 모든 CSV 탐색
    raw_files = sorted(raw_folder.glob("*.csv"))

    if not raw_files:
        raise ValueError(
            f"CSV 파일을 찾지 못했습니다: {raw_folder}"
        )

    print(f"발견한 Raw 파일: {len(raw_files)}개")

    # Raw CSV 결합
    raw_parts = []

    for file in raw_files:
        part = pd.read_csv(
            file,
            usecols=all_columns
        )

        if not part.empty:
            raw_parts.append(part)

    if not raw_parts:
        raise ValueError("분석 가능한 데이터가 없습니다.")

    raw_df = pd.concat(
        raw_parts,
        ignore_index=True
    )

    print(f"결합된 Raw Data: {raw_df.shape}")

    # 자료형 통일
    raw_df["STAGE"] = raw_df["STAGE"].astype(str)

    numeric_raw_columns = (
        ["WAFER_ID", "CHAMBER"]
        + process_columns
        + wear_columns
    )

    for column in numeric_raw_columns:
        raw_df[column] = pd.to_numeric(
            raw_df[column],
            errors="coerce"
        )

    # 웨이퍼별 공정변수 평균
    process_mean = (
        raw_df
        .groupby(["WAFER_ID", "STAGE"])[process_columns]
        .mean()
        .add_suffix("_mean")
    )

    # 웨이퍼별 공정변수 표준편차
    process_std = (
        raw_df
        .groupby(["WAFER_ID", "STAGE"])[process_columns]
        .std()
        .add_suffix("_std")
    )

    # 웨이퍼별 소모품 사용량 평균
    wear_mean = (
        raw_df
        .groupby(["WAFER_ID", "STAGE"])[wear_columns]
        .mean()
        .add_suffix("_mean")
    )

    # 웨이퍼가 사용한 챔버
    chamber = (
        raw_df
        .groupby(["WAFER_ID", "STAGE"])["CHAMBER"]
        .first()
    )

    # 모든 특징 결합
    features = pd.concat(
        [
            chamber,
            process_mean,
            process_std,
            wear_mean
        ],
        axis=1
    ).reset_index()

    # 모델 입력 형식으로 정렬
    X_new = features.reindex(
        columns=feature_columns
    ).copy()

    X_new["STAGE"] = X_new["STAGE"].astype(str)

    for column in X_new.columns:
        if column != "STAGE":
            X_new[column] = pd.to_numeric(
                X_new[column],
                errors="coerce"
            ).astype("float64")

    # 제거율 예측
    predictions = model.predict(X_new)

    result = features[
        ["WAFER_ID", "STAGE", "CHAMBER"]
    ].copy()

    result["PREDICTED_REMOVAL_RATE"] = predictions

    # 결과 저장
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"예측한 웨이퍼: {len(result)}개")
    print(f"결과 저장 완료: {output_path}")

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "사용법: python predict_cmp.py "
            "<Raw CSV 폴더> <결과 CSV 경로>"
        )
        sys.exit(1)

    predict_cmp_folder(
        raw_folder=sys.argv[1],
        output_path=sys.argv[2]
    )