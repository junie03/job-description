"""
분석 질문 2 (최종 기획안): 신입·주니어(최소 요구 경력 0~2년)·미들(3~5년)·시니어(6년 이상) 공고의
요구 역량은 직군별로 어떻게 다른가?

입력: data/processed/postings.csv
대상: 직군 라벨이 있는 공고(풀스택 경계 제외), 유사 공고 그룹당 1건, 경력 구간을 알 수 있는 공고

방법 (기획안 '분석 방법')
 1. 직군 × 경력 구간 분포
 2. 경력 구간별 가중 log-odds(해당 구간 vs 나머지 구간), 직군 × 경력 구간 log-odds(직군 안에서 비교)
 3. 역량 개념(CS 기본기·설계·운영·대용량·리딩·학습) 언급률
 4. 직군을 통제한 로지스틱 회귀: 개념 언급 여부 ~ 경력 구간(기준: 신입·주니어) + 직군
 5. 보조: 경력 구간별 공고당 요구 기술 수와 기술 언급 비율

출력: results/analysis2/*.csv, *.png, summary.txt
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, norm
from sklearn.linear_model import LogisticRegression

from tech_dict import TOKEN

OUT = Path("results/analysis2")
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = ["백엔드/서버개발", "프론트엔드", "인프라", "데이터엔지니어"]
BANDS = ["신입·주니어", "미들", "시니어"]
COLOR = {"백엔드/서버개발": "#2a78d6", "프론트엔드": "#eb6834", "인프라": "#1baf7a", "데이터엔지니어": "#eda100"}
# 경력 구간은 순서가 있는 값 → 같은 파랑의 밝기(연함 → 진함)
BAND_C = {"신입·주니어": "#9ec5f4", "미들": "#3987e5", "시니어": "#104281"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

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


def split(v):
    return [t for t in str(v).split("|") if t and t != "nan"]


TOK2CANON = {v: k for k, v in TOKEN.items()}
tech_tok = set(TOKEN.values())
ROLE_WORDS = {"백엔드", "backend", "프론트엔드", "프론트", "frontend", "인프라", "infra", "개발자", "엔지니어"}

# 역량 개념 사전(원문 텍스트에서 공고당 언급 여부로 셈)
CONCEPTS = {
    "CS 기본기": r"자료\s*구조|알고리즘|algorithm|운영\s*체제|컴퓨터\s*(?:공학|과학)|전산학|\bcs\b|객체\s*지향|디자인\s*패턴|네트워크\s*(?:기초|이해)|기본기",
    "설계": r"설계|아키텍처|architecture|아키텍쳐|모델링|구조\s*(?:개선|설계)",
    "운영": r"운영|장애|트러블\s*슈팅|troubleshoot|모니터링|유지\s*보수|on-?call|안정화",
    "대용량": r"대용량|대규모|트래픽|고가용성|확장성|분산\s*(?:처리|시스템|환경)|성능\s*(?:개선|최적화|튜닝)|동시성|scalab",
    # '주도적으로'는 태도 표현이라 리딩에서 뺌(KWIC로 확인), 팀·프로젝트를 이끄는 표현만 셈
    "리딩": r"리딩|리드(?!\s*타임)|\blead|멘토링(?!\s*(?:아래|을\s*받|받|제공|지원))|팀장|파트장|팀\s*리더|의사\s*결정|기술\s*방향|매니징|이끌|\bpl\b|기술\s*리더",
    "학습": r"학습|배우|성장|관심|열정|자기\s*주도|공부|호기심|습득|도전",
}

# ---------------- 데이터 ----------------
df = pd.read_csv("data/processed/postings.csv", dtype={"id": str})
df = df[(~df["is_boundary"]) & df["label_ko"].isin(GROUPS) & df["is_rep"] & df["exp_band"].isin(BANDS)].reset_index(drop=True)
df["tech_all"] = df["tech_all"].map(split)
df["tokens"] = df["tokens"].fillna("")
df["body"] = df[["duties", "reqs", "prefs"]].fillna("").agg("\n".join, axis=1).str.lower()

# ---------------- 1. 분포 ----------------
ct = pd.crosstab(df["label_ko"], df["exp_band"]).reindex(index=GROUPS, columns=BANDS)
ct["합계"] = ct.sum(axis=1)
ct.loc["합계"] = ct.sum()
ct.to_csv(OUT / "1_group_x_band.csv", encoding="utf-8-sig")
say(f"분석 대상(풀스택 경계 제외, 유사 공고 그룹당 1건, 경력 구간 있음): {len(df)}건")
say("직군 × 경력 구간:\n" + ct.to_string())
say("경력 구간 기준: 최소 요구 경력 0~2년 신입·주니어 / 3~5년 미들 / 6년 이상 시니어")

fig, ax = plt.subplots(figsize=(9, 3.3))
y = np.arange(len(GROUPS))[::-1]
left = np.zeros(len(GROUPS))
for b in BANDS:
    v = ct.loc[GROUPS, b].values.astype(float)
    p = v / ct.loc[GROUPS, "합계"].values
    ax.barh(y, p, left=left, color=BAND_C[b], edgecolor=SURFACE, linewidth=2, height=0.62, label=b)
    for yi, l0, pp, n in zip(y, left, p, v):
        ax.text(l0 + pp / 2, yi, f"{int(n)}건", ha="center", va="center", fontsize=9,
                color="white" if b != "신입·주니어" else INK)
    left += p
ax.set_yticks(y, [f"{g} (n={int(ct.loc[g, '합계'])})" for g in GROUPS])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False, fontsize=9)
ax.set_title("직군별 경력 구간 구성", loc="left", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig1_group_x_band.png", dpi=160)
plt.close(fig)


# ---------------- 2. log-odds ----------------
def weighted_log_odds(counts):
    vocab = sorted(set().union(*[c.keys() for c in counts.values()]))
    total = Counter()
    for c in counts.values():
        total.update(c)
    a = np.array([total[w] for w in vocab], float)
    a0 = a.sum()
    out = {}
    for g, c in counts.items():
        yi = np.array([c[w] for w in vocab], float)
        yj = a - yi
        ni, nj = yi.sum(), yj.sum()
        d = np.log((yi + a) / (ni + a0 - yi - a)) - np.log((yj + a) / (nj + a0 - yj - a))
        out[g] = pd.Series(d / np.sqrt(1 / (yi + a) + 1 / (yj + a)), index=vocab)
    return pd.DataFrame(out)


def doc_terms(s):
    """공고당 1회: 기술은 표준 표기로, 일반 단어는 그대로(직군 이름 제외)."""
    out = set()
    for w in s.split():
        if w in tech_tok:
            out.add(TOK2CANON[w])
        elif w not in ROLE_WORDS:
            out.add(w)
    return out


df["terms"] = df["tokens"].map(doc_terms)
tot = Counter(t for ts in df["terms"] for t in ts)
keep = {t for t, n in tot.items() if n >= 5}
lo_band = weighted_log_odds({b: Counter(t for ts in df[df["exp_band"] == b]["terms"] for t in ts if t in keep) for b in BANDS})
lo_band.round(2).to_csv(OUT / "2_log_odds_band.csv", encoding="utf-8-sig")
say("\n[2] 경력 구간별 특징 단어·기술 (가중 log-odds z, 해당 구간 vs 나머지 구간, 4개 직군 합산)")
for b in BANDS:
    say(f"  {b}: " + ", ".join(f"{t}({z:.1f})" for t, z in lo_band[b].sort_values(ascending=False).head(12).items()))

rows = []
say("  직군 × 경력 구간 (직군 안에서 해당 구간 vs 나머지 구간, 상위 6)")
for g in GROUPS:
    sub = df[df["label_ko"] == g]
    tg = Counter(t for ts in sub["terms"] for t in ts)
    kg = {t for t, n in tg.items() if n >= 3}
    lo = weighted_log_odds({b: Counter(t for ts in sub[sub["exp_band"] == b]["terms"] for t in ts if t in kg) for b in BANDS})
    for b in BANDS:
        top = lo[b].sort_values(ascending=False).head(10)
        for t, z in top.items():
            rows.append({"직군": g, "경력 구간": b, "단어·기술": t, "z": round(z, 2)})
        say(f"    {g} · {b}(n={int((sub['exp_band'] == b).sum())}): " + ", ".join(f"{t}({z:.1f})" for t, z in top.head(6).items()))
pd.DataFrame(rows).to_csv(OUT / "2_log_odds_group_x_band.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, b in zip(axes, BANDS):
    top = lo_band[b].sort_values(ascending=False).head(12)[::-1]
    ax.barh(range(len(top)), top.values, color=BAND_C[b], height=0.62)
    ax.set_yticks(range(len(top)), top.index)
    ax.tick_params(axis="y", length=0)
    ax.axvline(1.96, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{b} (n={int((df['exp_band'] == b).sum())})", loc="left", color=INK, fontsize=12)
    ax.set_xlabel("가중 log-odds z (점선 = 1.96)")
fig.suptitle("경력 구간별 특징 단어·기술 (해당 구간 vs 나머지 구간)", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig2_log_odds_band.png", dpi=160)
plt.close(fig)

# ---------------- 3. 역량 개념 언급률 ----------------
for k, rx in CONCEPTS.items():
    df[k] = df["body"].str.contains(rx, regex=True).astype(int)
rate = df.groupby(["label_ko", "exp_band"])[list(CONCEPTS)].mean().mul(100).round(1)
rate_all = df.groupby("exp_band")[list(CONCEPTS)].mean().mul(100).round(1).reindex(BANDS)
rate.to_csv(OUT / "3_concept_rate_group_x_band.csv", encoding="utf-8-sig")
rate_all.to_csv(OUT / "3_concept_rate_band.csv", encoding="utf-8-sig")
say("\n[3] 역량 개념 언급률(%) — 4개 직군 합산")
say(rate_all.to_string())

fig, axes = plt.subplots(1, len(CONCEPTS), figsize=(16, 4.6), sharey=True)
x = np.arange(len(GROUPS))
w = 0.26
for ax, k in zip(axes, CONCEPTS):
    for i, b in enumerate(BANDS):
        vals = [rate.loc[(g, b), k] if (g, b) in rate.index else np.nan for g in GROUPS]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=BAND_C[b], label=b)
    ax.set_xticks(x, [g.replace("/서버개발", "").replace("엔지니어", "") for g in GROUPS], fontsize=8.5, rotation=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)
    ax.set_title(k, loc="left", color=INK, fontsize=12)
axes[0].set_ylabel("언급 공고 비율(%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.93), frameon=False, fontsize=9.5)
fig.suptitle("역량 개념 언급률: 직군 × 경력 구간", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(OUT / "fig3_concept_rates.png", dpi=160)
plt.close(fig)


# ---------------- 4. 직군 통제 로지스틱 회귀 ----------------
def logit_wald(X, y, names):
    """비벌점 로지스틱 회귀 + Wald 검정(공분산 = 피셔 정보행렬의 역행렬)."""
    m = LogisticRegression(penalty=None, max_iter=5000, fit_intercept=True).fit(X, y)
    beta = np.r_[m.intercept_, m.coef_.ravel()]
    Xc = np.c_[np.ones(len(X)), X]
    p = 1 / (1 + np.exp(-Xc @ beta))
    cov = np.linalg.pinv(Xc.T @ (Xc * (p * (1 - p))[:, None]))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * (1 - norm.cdf(np.abs(z)))
    return pd.DataFrame({"coef": beta, "se": se, "p": pv}, index=["절편"] + names)


Xd = pd.get_dummies(df[["exp_band", "label_ko"]].astype("category"))
cols = ["exp_band_미들", "exp_band_시니어", "label_ko_프론트엔드", "label_ko_인프라", "label_ko_데이터엔지니어"]  # 기준: 신입·주니어, 백엔드
X = Xd[cols].astype(float).values
names = ["미들(vs 신입·주니어)", "시니어(vs 신입·주니어)", "프론트엔드(vs 백엔드)", "인프라(vs 백엔드)", "데이터엔지니어(vs 백엔드)"]
rows = []
say("\n[4] 직군을 통제한 로지스틱 회귀: 개념 언급 ~ 경력 구간 + 직군 (오즈비, 95% CI, p)")
for k in CONCEPTS:
    r = logit_wald(X, df[k].values, names)
    for nm in names:
        b, se, p = r.loc[nm]
        rows.append({"개념": k, "항": nm, "오즈비": np.exp(b), "CI_하한": np.exp(b - 1.96 * se), "CI_상한": np.exp(b + 1.96 * se), "p": p})
    mid, sen = r.loc[names[0]], r.loc[names[1]]
    star = lambda p: "**" if p < 0.01 else ("*" if p < 0.05 else "")
    say(f"  {k}: 미들 OR {np.exp(mid['coef']):.2f}{star(mid['p'])} (p={mid['p']:.3f}), "
        f"시니어 OR {np.exp(sen['coef']):.2f}{star(sen['p'])} (p={sen['p']:.3f})")
lr = pd.DataFrame(rows)
lr.round(4).to_csv(OUT / "4_logit_concepts.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(9, 5.2))
ks = list(CONCEPTS)
for j, (term, b) in enumerate([(names[0], "미들"), (names[1], "시니어")]):
    s = lr[lr["항"] == term].set_index("개념").reindex(ks)
    yy = np.arange(len(ks))[::-1] + (0.15 if j == 0 else -0.15)
    ax.hlines(yy, s["CI_하한"], s["CI_상한"], color=BAND_C[b], linewidth=2)
    ax.scatter(s["오즈비"], yy, s=60, color=BAND_C[b], edgecolor=SURFACE, linewidth=2, zorder=3, label=f"{b} vs 신입·주니어")
    for yi, (k, rr) in zip(yy, s.iterrows()):
        if rr["p"] < 0.05:
            ax.text(rr["CI_상한"] * 1.08, yi, "*" if rr["p"] >= 0.01 else "**", va="center", color=INK, fontsize=11)
ax.axvline(1, color=INK2, linewidth=0.9)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
ax.set_yticks(np.arange(len(ks))[::-1], ks)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("오즈비 (로그 척도, 직군 통제) · 1보다 크면 신입·주니어보다 더 많이 언급")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("경력 구간에 따른 역량 개념 언급 오즈비 (95% CI, * p<0.05, ** p<0.01)", loc="left", color=INK, fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT / "fig4_logit_odds_ratios.png", dpi=160)
plt.close(fig)

# ---------------- 5. 기술 수·기술 비율 ----------------
df["n_tech"] = df["tech_all"].str.len()
nt = df.groupby(["label_ko", "exp_band"])["n_tech"].mean().unstack().reindex(index=GROUPS, columns=BANDS).round(2)
p_kw = {g: kruskal(*[df[(df["label_ko"] == g) & (df["exp_band"] == b)]["n_tech"] for b in BANDS]).pvalue for g in GROUPS}
nt["Kruskal-Wallis p"] = pd.Series(p_kw).round(3)
nt.to_csv(OUT / "5_n_tech_by_band.csv", encoding="utf-8-sig")
say("\n[5] 공고당 요구 기술 수 평균 (직군 × 경력 구간)")
say(nt.to_string())

techs = [t for t, n in Counter(t for ts in df["tech_all"] for t in ts).most_common() if n >= 15]
rows = []
for g in GROUPS:
    for t in techs:
        r = {"직군": g, "기술": t}
        for b in BANDS:
            sub = df[(df["label_ko"] == g) & (df["exp_band"] == b)]
            r[f"{b}(%)"] = round(100 * sum(t in ts for ts in sub["tech_all"]) / max(len(sub), 1), 1)
        r["시니어-신입(%p)"] = round(r["시니어(%)"] - r["신입·주니어(%)"], 1)
        rows.append(r)
tb = pd.DataFrame(rows)
tb.to_csv(OUT / "5_tech_share_by_band.csv", index=False, encoding="utf-8-sig")
say("  직군별로 시니어에서 더 많이(+) / 신입·주니어에서 더 많이(−) 언급되는 기술 (차이 상위 4, 표본이 작아 경향 수준)")
for g in GROUPS:
    s = tb[tb["직군"] == g].sort_values("시니어-신입(%p)")
    say(f"    {g}: 시니어↑ " + ", ".join(f"{t}(+{v:.0f})" for t, v in s.tail(4)[::-1][["기술", "시니어-신입(%p)"]].values) +
        " / 신입↑ " + ", ".join(f"{t}({v:.0f})" for t, v in s.head(4)[["기술", "시니어-신입(%p)"]].values))

(OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
print(f"\n저장 위치: {OUT}")
