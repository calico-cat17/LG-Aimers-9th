# LG Aimers 9기 × LG 트윈스 해커톤 — 투구 제구 성공 확률 예측

투구 직전까지 확인 가능한 정보만으로 다음 투구의 제구 성공 확률(`control_success`)을
예측하는 이진분류 문제. 평가지표는 Brier Skill Score(BSS).

## 폴더 구조

```
repo/
├── src/            # 재사용 모듈 (feature engineering, 학습, 리포트, 행독립성 감사 도구)
├── final/
│   
│   
├── archive/        # Phase별 1회성 실험 스크립트 (채택/기각 여부와 무관하게 시도 기록으로 보존)
│
└── docs/           # 설계 배경, 점수 분석, 다음 전략, 용어집 등 문서
```

## 핵심 결과

| 제출 | 실제 리더보드 점수 |
|---|---|


## 재현 방법


## 데이터

`train.csv`, `test.csv`, `trackman_history.csv` 등 대회 제공 데이터와 대회 측 baseline
(`baseline_submit/`)은 저장소에 포함하지 않음(`.gitignore` 참고). 데이터는 데이콘 대회
페이지에서 직접 받아야 함.
