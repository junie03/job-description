# 채용공고 텍스트 분석을 통한 IT 직군별 요구 역량 비교

2026-2학기 텍스트데이터분석 프로젝트 (4조: 권태현, 이주표)

점핏과 사람인의 IT 채용공고(주요업무·자격요건·우대사항)를 수집해 **백엔드/서버개발·프론트엔드·인프라·데이터엔지니어** 4개 직군의 요구 역량을 비교한다.

| 분석 질문 | 방법 | 주요 결과 |
|---|---|---|
| 1. 직군별 기술 스택과 우대 조건 | 기술 언급 비율, 자격요건 vs 우대사항, TF-IDF, 가중 log-odds, NPMI 공출현 네트워크 + Louvain, K-means·Ward 군집, 기술 덴드로그램, KWIC | 직군별 기술이 뚜렷하게 갈림(K-means ARI 0.70) |
| 2. 경력 구간(신입·주니어 / 미들 / 시니어)별 요구 역량 | 구간별 log-odds, 역량 개념 6개 언급률, 직군 통제 로지스틱 회귀 | 경력이 오를수록 설계·대용량·리딩 증가, 학습 감소 |
| 3. 텍스트만으로 직군 분류 | TF-IDF + 로지스틱 회귀·선형 SVM, 층화 GroupKFold 5-fold × 시드 5회, 교차 출처 실험, 오분류 원문 검토 | macro F1 0.93 (기준선 0.13~0.28) |

## 데이터

- 수집: 2026-11-23, 점핏 348건 + 사람인 241건 = 589건
- 정제 후: 330건 (백엔드/서버개발 94, 인프라 77, 데이터엔지니어 51, 프론트엔드 48 + 풀스택 경계 60)
- 채용공고 원문은 수업 분석 용도로만 사용한다.

## 파일 구성

| 경로 | 내용 |
|---|---|
| `crawl.py` | 점핏·사람인 수집 (robots.txt 허용 경로, 요청 간 1~1.5초 대기, 이어받기 지원) |
| `tech_dict.py` | 기술 용어 사전 (표준어 268개, 표기 변형 통합) |
| `preprocess.py` | 본문 점검, 직군 판정, 정제, Kiwi 형태소 분석, 유사 공고 그룹, 경력 구간 |
| `analysis1.py` ~ `analysis3.py` | 분석 질문 1~3 |
| `data/raw/` | 수집 원자료 (`jumpit.jsonl`, `saramin.jsonl`) |
| `data/processed/` | 분석용 데이터 `postings.csv`, 전처리 요약, 사전 점검 결과 |
| `data/manual/` | 규칙으로 판정하기 어려운 공고의 원문 판정(`manual_labels.csv`)과 오분류 유형(`misclassified_review.csv`) |
| `results/analysis1~3/` | 분석별 표(csv)·그림(png)·요약(`summary.txt`) |
| `최종보고서.pdf` | 최종 보고서 |

## 실행

```bash
pip install -r requirements.txt
python crawl.py          # 수집 (약 15분, data/raw/에 이미 결과가 있으면 건너뜀)
python preprocess.py     # 전처리 → data/processed/postings.csv
python analysis1.py      # 결과 → results/analysis1/
python analysis2.py
python analysis3.py
```

그림은 Windows의 `Malgun Gothic` 글꼴을 사용한다. 다른 OS에서는 각 분석 코드의 `font.family`를 바꾼다.
