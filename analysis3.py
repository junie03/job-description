"""
분석 질문 3 (최종 기획안): 공고 텍스트만으로 4개 직군을 분류할 수 있으며,
어떤 공고(풀스택, DevOps/SRE, 데이터 플랫폼)가 직군 경계에서 혼동되는가?

입력: data/processed/postings.csv (+ 오분류 원문 검토 결과 data/manual/misclassified_review.csv가 있으면 병합)
학습·평가: 직군 라벨이 있는 공고 전체(풀스택 경계 제외). 유사 공고 그룹(sim_group)은 같은 폴드에만 배정
입력 텍스트: 주요업무·자격요건·우대사항 토큰(기술 표준 토큰 + Kiwi 명사), 제목은 쓰지 않음

방법 (기획안 '분석 방법'·'평가 방법')
 1. TF-IDF + 로지스틱 회귀 / 선형 SVM
 2. 층화 GroupKFold 5-fold × 시드 5회, 정확도·macro F1, 기준선(최빈 클래스·층화 무작위)과 비교
 3. 추가 점검: 직군명 단어를 뺀 모델, 기술 용어만 쓴 모델, 교차 출처(점핏→사람인, 사람인→점핏)
 4. 혼동행렬(5회 반복의 폴드 밖 예측 확률 평균), 분류 기여 단어
 5. 오분류 공고 목록 → 원문 검토 유형(경계 공고·라벨 오류·본문 부족·모델 오류)
 6. 풀스택 경계 공고: 학습에서 빼고 예측해 백엔드/프론트엔드 기울기 확인

출력: results/analysis3/*.csv, *.png, summary.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from tech_dict import TOKEN

OUT = Path("results/analysis3")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]

GROUPS = ["백엔드/서버개발", "프론트엔드", "인프라", "데이터엔지니어"]
SHORT = {"백엔드/서버개발": "백엔드", "프론트엔드": "프론트엔드", "인프라": "인프라", "데이터엔지니어": "데이터"}
COLOR = {"백엔드/서버개발": "#2a78d6", "프론트엔드": "#eb6834", "인프라": "#1baf7a", "데이터엔지니어": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
MUTED, ACCENT = "#b9b8b2", "#3987e5"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
summary = []


def say(s=""):
    print(s)
    summary.append(s)


TOK2CANON = {v: k for k, v in TOKEN.items()}
TECH_TOK = set(TOKEN.values())
# 직군명 단어(정답을 거의 그대로 알려 주는 단어). DevOps·SRE는 기술 사전에도 있지만 직무명이므로 함께 뺌
ROLE = {"백엔드", "backend", "back", "프론트엔드", "프론트", "frontend", "front", "인프라", "infra", "데브옵스",
        "데이터엔지니어", "엔지니어링", "풀스택", "fullstack", "서버개발", TOKEN["DevOps"], TOKEN["SRE"]}

# ---------------- 데이터 ----------------
allp = pd.read_csv("data/processed/postings.csv", dtype={"id": str})
allp["tokens"] = allp["tokens"].fillna("")
core = allp[(~allp["is_boundary"]) & allp["label_ko"].isin(GROUPS)].reset_index(drop=True)
bound = allp[allp["is_boundary"]].reset_index(drop=True)
y = core["label_ko"].to_numpy(dtype=object)
groups = core["sim_group"].to_numpy()

TEXT = {
    "전체 토큰": lambda s: s,
    "직군명 단어 제거": lambda s: " ".join(w for w in s.split() if w not in ROLE),
    "기술 용어만": lambda s: " ".join(w for w in s.split() if w in TECH_TOK and w not in ROLE),
}
say(f"학습·평가 공고: {len(core)}건 ({', '.join(f'{g} {(y == g).sum()}' for g in GROUPS)}), 유사 공고 그룹 {core['sim_group'].nunique()}개")
say(f"출처: {core['source'].value_counts().to_dict()} / 풀스택 경계 공고(예측만): {len(bound)}건")


def tfidf():
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2, max_df=0.8, sublinear_tf=True)


MODELS = {
    "기준선: 최빈 클래스": lambda s: DummyClassifier(strategy="most_frequent"),
    "기준선: 층화 무작위": lambda s: DummyClassifier(strategy="stratified", random_state=s),
    "로지스틱 회귀": lambda s: LogisticRegression(C=5, class_weight="balanced", max_iter=5000, random_state=s),
    "선형 SVM": lambda s: LinearSVC(C=0.5, class_weight="balanced", random_state=s),
}


def run_cv(text_fn, model_name, return_proba=False):
    X = core["tokens"].map(text_fn)
    accs, f1s = [], []
    proba_sum = np.zeros((len(core), 4))
    for s in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
        pred = np.empty(len(core), dtype=object)
        for tr, te in cv.split(X, y, groups):
            m = make_pipeline(tfidf(), MODELS[model_name](s)).fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
            if return_proba:
                cls = list(m.classes_)
                proba_sum[te] += m.predict_proba(X.iloc[te])[:, [cls.index(g) for g in GROUPS]]
        accs.append(accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, average="macro"))
    out = {"정확도": np.mean(accs), "정확도_sd": np.std(accs), "macro F1": np.mean(f1s), "macro F1_sd": np.std(f1s)}
    return (out, proba_sum / len(SEEDS)) if return_proba else out


# ---------------- 1~2. 모델 비교 ----------------
rows = []
for name in MODELS:
    rows.append({"모델": name, "특성": "전체 토큰", **run_cv(TEXT["전체 토큰"], name)})
# ---------------- 3. 추가 점검 ----------------
for feat in ("직군명 단어 제거", "기술 용어만"):
    for name in ("로지스틱 회귀", "선형 SVM"):
        rows.append({"모델": name, "특성": feat, **run_cv(TEXT[feat], name)})
res = pd.DataFrame(rows)
res.round(3).to_csv(OUT / "1_model_comparison.csv", index=False, encoding="utf-8-sig")
say("\n[1~3] 층화 GroupKFold 5-fold × 시드 5회 (평균 ± 표준편차)")
for _, r in res.iterrows():
    say(f"  {r['모델']:<14} [{r['특성']}] 정확도 {r['정확도']:.3f}±{r['정확도_sd']:.3f}, macro F1 {r['macro F1']:.3f}±{r['macro F1_sd']:.3f}")

# 교차 출처: 한 사이트로 학습 → 다른 사이트에서 평가 (유사 공고가 양쪽에 걸치지 않도록 상대 쪽과 같은 그룹은 학습에서 뺌)
rows = []
for tr_src, te_src in (("jumpit", "saramin"), ("saramin", "jumpit")):
    te_mask = (core["source"] == te_src).values
    tr_mask = (core["source"] == tr_src).values & ~core["sim_group"].isin(core.loc[te_mask, "sim_group"]).values
    for feat in ("전체 토큰", "직군명 단어 제거"):
        X = core["tokens"].map(TEXT[feat])
        m = make_pipeline(tfidf(), MODELS["로지스틱 회귀"](0)).fit(X[tr_mask], y[tr_mask])
        p = m.predict(X[te_mask])
        per = precision_recall_fscore_support(y[te_mask], p, labels=GROUPS, zero_division=0)[2]
        rows.append({"학습": tr_src, "평가": te_src, "특성": feat, "학습 n": int(tr_mask.sum()), "평가 n": int(te_mask.sum()),
                     "정확도": accuracy_score(y[te_mask], p), "macro F1": f1_score(y[te_mask], p, average="macro"),
                     **{f"F1_{SHORT[g]}": f for g, f in zip(GROUPS, per)},
                     "학습쪽 직군 분포": str(pd.Series(y[tr_mask]).value_counts().reindex(GROUPS).fillna(0).astype(int).to_dict())})
xs = pd.DataFrame(rows)
xs.round(3).to_csv(OUT / "3_cross_source.csv", index=False, encoding="utf-8-sig")
say("\n[3] 교차 출처 실험 (로지스틱 회귀)")
for _, r in xs.iterrows():
    say(f"  {r['학습']}→{r['평가']} [{r['특성']}] (학습 {r['학습 n']}, 평가 {r['평가 n']}): 정확도 {r['정확도']:.3f}, macro F1 {r['macro F1']:.3f} "
        f"| 직군별 F1 " + ", ".join(f"{SHORT[g]} {r['F1_' + SHORT[g]]:.2f}" for g in GROUPS))

# 그림 1: 모델·특성 비교
fig, ax = plt.subplots(figsize=(10, 4.8))
lab = [f"{m} · {f}" if not m.startswith("기준선") else m for m, f in zip(res["모델"], res["특성"])]
yy = np.arange(len(res))[::-1]
colors = [MUTED if m.startswith("기준선") else ACCENT for m in res["모델"]]
ax.barh(yy, res["macro F1"], xerr=res["macro F1_sd"], color=colors, height=0.6,
        error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
for yi, v, sd in zip(yy, res["macro F1"], res["macro F1_sd"]):
    ax.text(v + sd + 0.015, yi, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(yy, lab)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.08)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("macro F1 (층화 GroupKFold 5-fold × 시드 5회 평균, 막대 끝선 = 표준편차)")
ax.set_title("모델·특성별 직군 분류 성능", loc="left", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig1_model_comparison.png", dpi=160)
plt.close(fig)

# ---------------- 4. 혼동행렬·기여 단어 (주 모델: 로지스틱 회귀, 전체 토큰) ----------------
_, proba = run_cv(TEXT["전체 토큰"], "로지스틱 회귀", return_proba=True)
core["pred"] = np.array(GROUPS, dtype=object)[proba.argmax(axis=1)]
core["max_prob"] = proba.max(axis=1)
for j, g in enumerate(GROUPS):
    core[f"p_{SHORT[g]}"] = proba[:, j].round(3)
cm = confusion_matrix(y, core["pred"], labels=GROUPS)
pd.DataFrame(cm, index=[f"실제_{g}" for g in GROUPS], columns=[f"예측_{g}" for g in GROUPS]).to_csv(OUT / "4_confusion_matrix.csv", encoding="utf-8-sig")
prf = precision_recall_fscore_support(y, core["pred"], labels=GROUPS, zero_division=0)
say("\n[4] 혼동행렬 기준 직군별 성능 (5회 반복 폴드 밖 확률 평균의 argmax)")
for g, p, r, f, n in zip(GROUPS, *prf):
    say(f"  {g}: 정밀도 {p:.2f}, 재현율 {r:.2f}, F1 {f:.2f} (n={n})")
off = [(GROUPS[i], GROUPS[j], cm[i, j]) for i in range(4) for j in range(4) if i != j and cm[i, j] > 0]
say("  주요 혼동(실제→예측): " + ", ".join(f"{SHORT[a]}→{SHORT[b]} {n}건" for a, b, n in sorted(off, key=lambda x: -x[2])[:6]))

cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.2, 5.6))
im = ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQ), vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=9.5, color="white" if cmn[i, j] >= 0.5 else INK)
ax.set_xticks(range(4), GROUPS)
ax.set_yticks(range(4), GROUPS)
ax.set_xlabel("예측 직군")
ax.set_ylabel("실제 직군")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, 4), minor=True)
ax.set_yticks(np.arange(-.5, 4), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, format=matplotlib.ticker.PercentFormatter(1))
cb.outline.set_visible(False)
ax.set_title("혼동행렬 (로지스틱 회귀, 행 기준 비율·괄호 = 건수)", loc="left", color=INK, fontsize=12)
fig.tight_layout()
fig.subplots_adjust(left=0.2)
fig.savefig(OUT / "fig2_confusion_matrix.png", dpi=160)
plt.close(fig)

final = make_pipeline(tfidf(), MODELS["로지스틱 회귀"](0)).fit(core["tokens"], y)
vec, clf = final.steps[0][1], final.steps[1][1]
terms = [TOK2CANON.get(t, t) for t in vec.get_feature_names_out()]
coef = pd.DataFrame(clf.coef_.T, index=terms, columns=clf.classes_)[GROUPS]
coef.round(3).to_csv(OUT / "4_coefficients.csv", encoding="utf-8-sig")
say("  분류 기여 단어(로지스틱 회귀 계수 상위 10, 전체 토큰 모델)")
for g in GROUPS:
    say(f"    {g}: " + ", ".join(coef[g].sort_values(ascending=False).head(10).index))
final_nr = make_pipeline(tfidf(), MODELS["로지스틱 회귀"](0)).fit(core["tokens"].map(TEXT["직군명 단어 제거"]), y)
v2, c2 = final_nr.steps[0][1], final_nr.steps[1][1]
coef_nr = pd.DataFrame(c2.coef_.T, index=[TOK2CANON.get(t, t) for t in v2.get_feature_names_out()], columns=c2.classes_)[GROUPS]
coef_nr.round(3).to_csv(OUT / "4_coefficients_no_role.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8))
for ax, g in zip(axes, GROUPS):
    top = coef_nr[g].sort_values(ascending=False).head(12)[::-1]
    ax.barh(range(len(top)), top.values, color=COLOR[g], height=0.62)
    ax.set_yticks(range(len(top)), top.index)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(g, loc="left", color=INK, fontsize=12)
    ax.set_xlabel("회귀계수")
fig.suptitle("직군 분류에 기여한 단어·기술 (직군명 단어를 뺀 로지스틱 회귀, 계수 상위 12)", x=0.01, ha="left", fontsize=13.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig3_top_features.png", dpi=160)
plt.close(fig)

# ---------------- 5. 오분류 ----------------
mis = core[core["label_ko"] != core["pred"]].copy()
mis["혼동방향"] = mis["label_ko"].map(SHORT) + "→" + mis["pred"].map(SHORT)
mis["본문 토큰 수"] = mis["tokens"].str.split().str.len()
cols = ["id", "source", "url", "company", "title", "label_ko", "pred", "혼동방향", "max_prob"] + \
       [f"p_{SHORT[g]}" for g in GROUPS] + ["label_method", "본문 토큰 수", "tech_all", "duties", "reqs"]
mis = mis.sort_values(["혼동방향", "max_prob"], ascending=[True, False])
mis[cols].to_csv(OUT / "5_misclassified.csv", index=False, encoding="utf-8-sig")
say(f"\n[5] 오분류 {len(mis)}건 / {len(core)}건 ({len(mis) / len(core):.1%}) → 5_misclassified.csv")
rv = Path("data/manual/misclassified_review.csv")
if rv.exists():
    r = pd.read_csv(rv, dtype={"id": str})
    m = mis.merge(r[["id", "유형", "근거"]], on="id", how="left")
    m[cols[:8] + ["유형", "근거"]].to_csv(OUT / "5_misclassified_reviewed.csv", index=False, encoding="utf-8-sig")
    say("  원문 검토 유형: " + str(m["유형"].value_counts(dropna=False).to_dict()))
    say("  혼동 방향 × 유형:\n" + pd.crosstab(m["혼동방향"], m["유형"]).to_string())

# ---------------- 6. 풀스택 경계 공고 ----------------
pb = final.predict_proba(bound["tokens"])
cls = list(final.classes_)
pb = pb[:, [cls.index(g) for g in GROUPS]]
bound["pred"] = np.array(GROUPS, dtype=object)[pb.argmax(axis=1)]
bound["max_prob"] = pb.max(axis=1)
for j, g in enumerate(GROUPS):
    bound[f"p_{SHORT[g]}"] = pb[:, j].round(3)
bound["백엔드-프론트 확률차"] = (bound["p_백엔드"] - bound["p_프론트엔드"]).round(3)
bound[["id", "source", "title", "label_method", "pred", "max_prob"] + [f"p_{SHORT[g]}" for g in GROUPS] + ["백엔드-프론트 확률차"]] \
    .to_csv(OUT / "6_fullstack_predictions.csv", index=False, encoding="utf-8-sig")
dist = bound["pred"].value_counts().reindex(GROUPS).fillna(0).astype(int)
say(f"\n[6] 풀스택 경계 공고 {len(bound)}건 예측: " + ", ".join(f"{g} {n}건" for g, n in dist.items()))
say(f"  평균 확률: " + ", ".join(f"{SHORT[g]} {bound['p_' + SHORT[g]].mean():.2f}" for g in GROUPS) +
    f" / 확신도 중앙값 경계 {bound['max_prob'].median():.2f} vs 직군 공고(폴드 밖) {core['max_prob'].median():.2f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.3, 1]})
d = bound.sort_values("백엔드-프론트 확률차")
a1.scatter(d["백엔드-프론트 확률차"], np.arange(len(d)), s=28,
           c=[COLOR[g] for g in d["pred"]], edgecolor=SURFACE, linewidth=1)
a1.axvline(0, color=INK2, linewidth=0.9)
a1.set_yticks([])
a1.spines["left"].set_visible(False)
a1.set_xlim(-1, 1)
a1.set_xlabel("← 프론트엔드 쪽     P(백엔드) - P(프론트엔드)     백엔드 쪽 →")
for g in GROUPS:
    a1.scatter([], [], s=40, color=COLOR[g], label=f"예측 {g} ({dist[g]}건)")
a1.legend(loc="upper left", frameon=False, fontsize=8.5)
a1.xaxis.grid(True, color=GRID, linewidth=0.8)
a1.set_axisbelow(True)
a1.set_title(f"풀스택 경계 공고 {len(bound)}건의 기울기", loc="left", color=INK, fontsize=12)
bins = np.linspace(0.25, 1, 16)
for data, name, c in [(core["max_prob"], "직군 공고(폴드 밖 예측)", "#104281"), (bound["max_prob"], "풀스택 경계 공고", "#86b6ef")]:
    a2.hist(data, bins=bins, weights=np.ones(len(data)) / len(data), histtype="step", linewidth=2, color=c, label=name)
a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
a2.yaxis.grid(True, color=GRID, linewidth=0.8)
a2.set_axisbelow(True)
a2.set_xlabel("예측 확신도(최대 확률)")
a2.legend(loc="upper left", frameon=False, fontsize=9)
a2.set_title("예측 확신도 분포", loc="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_fullstack.png", dpi=160)
plt.close(fig)

(OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
print(f"\n저장 위치: {OUT}")
