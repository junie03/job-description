"""
분석 질문 1 (최종 기획안): 백엔드/서버개발·프론트엔드·인프라·데이터엔지니어 직군별로
요구하는 기술 스택과 우대 조건은 어떻게 다른가?

입력: data/processed/postings.csv
대상: 직군 라벨이 있는 공고(풀스택 경계 공고 제외), 유사 공고 그룹당 1건(is_rep)

방법 (기획안 '분석 방법' 그대로)
 1. 빈도: 직군별 기술 용어 언급 공고 비율
 2. 자격요건(필수) vs 우대사항(선택) 기술 언급 비율 비교
 3. TF-IDF: 직군별 평균 TF-IDF 상위어
 4. 가중 log-odds ratio(informative Dirichlet prior, Monroe et al., 2008): 한 직군 vs 나머지 직군, 단어·기술
 5. 기술 공출현 네트워크: 같은 공고에 함께 나온 기술 쌍을 NPMI로 거른 뒤 Louvain으로 기술 묶음 탐지
 6. 군집화: 공고 TF-IDF 벡터 K-means + 계층적 군집(Ward), 기술 용어 덴드로그램
 7. 검증: 상위 키워드의 원문 맥락(KWIC)

출력: results/analysis1/*.csv, *.png, summary.txt
"""
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

from tech_dict import TECH, TECH_CATEGORY, TOKEN, find_spans

OUT = Path("results/analysis1")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

GROUPS = ["백엔드/서버개발", "프론트엔드", "인프라", "데이터엔지니어"]
SHORT = {"백엔드/서버개발": "백엔드", "프론트엔드": "프론트엔드", "인프라": "인프라", "데이터엔지니어": "데이터"}
COLOR = {"백엔드/서버개발": "#2a78d6", "프론트엔드": "#eb6834", "인프라": "#1baf7a", "데이터엔지니어": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REQ_C, PREF_C = "#256abf", "#9ec5f4"
SEQ = ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

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


def show(tok):
    """토큰을 사람이 읽는 표기로(T_Spring_Boot → Spring Boot)."""
    return TOK2CANON.get(tok, tok)


# ---------------- 데이터 ----------------
df = pd.read_csv("data/processed/postings.csv", dtype={"id": str})
df = df[(~df["is_boundary"]) & df["label_ko"].isin(GROUPS) & df["is_rep"]].reset_index(drop=True)
for c in ("tech_all", "tech_reqs", "tech_prefs", "tech_duties"):
    df[c] = df[c].map(split)
for c in ("tokens", "tok_reqs", "tok_prefs"):
    df[c] = df[c].fillna("")
N = df["label_ko"].value_counts().reindex(GROUPS)
say("분석 대상(풀스택 경계 제외, 유사 공고 그룹당 1건): " + ", ".join(f"{g} {n}건" for g, n in N.items()) + f" / 합계 {N.sum()}건")
say("출처: " + str(df["source"].value_counts().to_dict()))
techs = sorted({t for ts in df["tech_all"] for t in ts})


def share_table(col):
    out = {}
    for g in GROUPS:
        sub = df[df["label_ko"] == g]
        c = Counter(t for ts in sub[col] for t in ts)
        out[g] = {t: 100 * c[t] / len(sub) for t in techs}
    return pd.DataFrame(out)


# ---------------- 1. 기술 언급 비율 ----------------
share = share_table("tech_all")
share.insert(0, "분류", [TECH_CATEGORY[t] for t in share.index])
share["전체"] = [100 * sum(t in ts for ts in df["tech_all"]) / len(df) for t in share.index]
share.round(1).sort_values("전체", ascending=False).to_csv(OUT / "1_tech_share.csv", encoding="utf-8-sig")
say("\n[1] 직군별 기술 언급 공고 비율 상위 10")
for g in GROUPS:
    say(f"  {g}: " + ", ".join(f"{t} {v:.0f}%" for t, v in share[g].sort_values(ascending=False).head(10).items()))
common = share[share[GROUPS].min(axis=1) >= 15].index.tolist()
say("  4개 직군 모두 15% 이상인 공통 기술: " + (", ".join(common) or "없음"))

top30 = share[GROUPS].max(axis=1).sort_values(ascending=False).head(30).index
order = sorted(top30, key=lambda t: (GROUPS.index(share.loc[t, GROUPS].astype(float).idxmax()), -share.loc[t, GROUPS].max()))
H = share.loc[order, GROUPS].astype(float)
fig, ax = plt.subplots(figsize=(7.4, 11))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQ)
im = ax.imshow(H.values, cmap=cmap, vmin=0, vmax=max(80, H.values.max()), aspect="auto")
ax.set_xticks(range(4), [f"{g}\n(n={N[g]})" for g in GROUPS])
ax.xaxis.tick_top()
ax.set_yticks(range(len(order)), order)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(len(order)):
    for j in range(4):
        v = H.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white" if v >= 45 else INK)
ax.set_xticks(np.arange(-.5, 4), minor=True)
ax.set_yticks(np.arange(-.5, len(order)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("해당 기술을 언급한 공고 비율(%)")
cb.outline.set_visible(False)
ax.set_title("직군별 기술 언급 비율 (상위 30개 기술)", loc="left", pad=48, fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig1_tech_share_heatmap.png", dpi=160)
plt.close(fig)

# ---------------- 2. 자격요건(필수) vs 우대사항(선택) ----------------
req, pref = share_table("tech_reqs"), share_table("tech_prefs")
rows = []
for g in GROUPS:
    for t in techs:
        r, p = req.loc[t, g], pref.loc[t, g]
        if r + p > 0:
            rows.append({"직군": g, "기술": t, "자격요건(%)": round(r, 1), "우대사항(%)": round(p, 1),
                         "우대_편중도": round(p / (r + p), 2)})
rp = pd.DataFrame(rows)
rp.to_csv(OUT / "2_reqs_vs_prefs.csv", index=False, encoding="utf-8-sig")
say("\n[2] 자격요건(필수) vs 우대사항(선택)")
for g in GROUPS:
    s = rp[rp["직군"] == g]
    must = s.sort_values("자격요건(%)", ascending=False).head(6)
    plus = s[(s["우대사항(%)"] >= 5) & (s["우대_편중도"] >= 0.6)].sort_values("우대사항(%)", ascending=False).head(6)
    say(f"  {g} 자격요건 상위: " + ", ".join(f"{t}({v:.0f}%)" for t, v in zip(must["기술"], must["자격요건(%)"])))
    say(f"  {g} 주로 우대사항에 나옴: " + (", ".join(f"{t}({v:.0f}%)" for t, v in zip(plus["기술"], plus["우대사항(%)"])) or "없음"))

fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.8), sharex=True)
for ax, g in zip(axes, GROUPS):
    s = rp[rp["직군"] == g].assign(tot=lambda d: d["자격요건(%)"] + d["우대사항(%)"]).sort_values("tot", ascending=False).head(12)[::-1]
    y = np.arange(len(s))
    ax.barh(y + 0.2, s["자격요건(%)"], height=0.38, color=REQ_C, label="자격요건(필수)")
    ax.barh(y - 0.2, s["우대사항(%)"], height=0.38, color=PREF_C, label="우대사항(선택)")
    ax.set_yticks(y, s["기술"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{g} (n={N[g]})", loc="left", color=INK, fontsize=12)
    ax.set_xlabel("언급 공고 비율(%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=2, frameon=False, fontsize=10)
fig.suptitle("직군별 상위 기술: 자격요건(필수) vs 우대사항(선택)", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "fig2_reqs_vs_prefs.png", dpi=160)
plt.close(fig)

# ---------------- 3. TF-IDF ----------------
vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=3, max_df=0.7, sublinear_tf=True)
X = vec.fit_transform(df["tokens"])
terms = vec.get_feature_names_out()
tfidf = pd.DataFrame({g: np.asarray(X[(df["label_ko"] == g).values].mean(axis=0)).ravel() for g in GROUPS}, index=terms)
tfidf.index = [show(t) for t in tfidf.index]
tfidf.round(4).to_csv(OUT / "3_tfidf_by_group.csv", encoding="utf-8-sig")
say("\n[3] 직군 평균 TF-IDF 상위 12")
for g in GROUPS:
    say(f"  {g}: " + ", ".join(tfidf[g].sort_values(ascending=False).head(12).index))


# ---------------- 4. 가중 log-odds ----------------
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


tech_tok = set(TOKEN.values())
# 직군 이름 자체는 '특징 단어'로서 정보가 없어 단어 log-odds에서 뺌(TF-IDF 표에는 그대로 둠)
ROLE_WORDS = {"백엔드", "backend", "프론트엔드", "프론트", "frontend", "인프라", "infra", "데이터엔지니어", "엔지니어링", "개발자", "엔지니어"}
# 공고당 1회(문서 빈도)로 세어 긴 공고 하나가 결과를 좌우하지 않게 함
tc = {g: Counter(t for ts in df[df["label_ko"] == g]["tech_all"] for t in ts) for g in GROUPS}
wc_raw = {g: Counter(w for s in df[df["label_ko"] == g]["tokens"] for w in set(s.split()) if w not in tech_tok and w not in ROLE_WORDS) for g in GROUPS}
tot_w = sum(wc_raw.values(), Counter())
wc = {g: Counter({w: n for w, n in c.items() if tot_w[w] >= 5}) for g, c in wc_raw.items()}
lo_t, lo_w = weighted_log_odds(tc), weighted_log_odds(wc)
lo_t.round(2).to_csv(OUT / "4_log_odds_tech.csv", encoding="utf-8-sig")
lo_w.round(2).to_csv(OUT / "4_log_odds_words.csv", encoding="utf-8-sig")
say("\n[4] 가중 log-odds z 상위 (z > 1.96 유의)")
for g in GROUPS:
    say(f"  {g} 기술: " + ", ".join(f"{t}({z:.1f})" for t, z in lo_t[g].sort_values(ascending=False).head(8).items()))
    say(f"  {g} 단어: " + ", ".join(f"{t}({z:.1f})" for t, z in lo_w[g].sort_values(ascending=False).head(10).items()))

fig, axes = plt.subplots(2, 4, figsize=(15.5, 9))
for row, (lo, kind) in enumerate([(lo_t, "기술"), (lo_w, "단어")]):
    for ax, g in zip(axes[row], GROUPS):
        top = lo[g].sort_values(ascending=False).head(10)[::-1]
        ax.barh(range(len(top)), top.values, color=COLOR[g], height=0.62)
        ax.set_yticks(range(len(top)), top.index)
        ax.tick_params(axis="y", length=0)
        ax.axvline(1.96, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{g} · 특징 {kind}", loc="left", color=INK, fontsize=11.5)
        if row == 1:
            ax.set_xlabel("가중 log-odds z (점선 = 1.96)")
fig.suptitle("직군 특징 기술·단어 (해당 직군 vs 나머지 3개 직군)", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "fig3_log_odds.png", dpi=160)
plt.close(fig)

# ---------------- 5. 공출현 네트워크 (NPMI + Louvain) ----------------
n_doc = len(df)
single = Counter(t for ts in df["tech_all"] for t in ts)
pairs = Counter(p for ts in df["tech_all"] for p in combinations(sorted(set(ts)), 2))
NPMI_MIN, MIN_PAIR, MIN_TECH = 0.1, 5, 8
rows = []
for (a, b), c in pairs.items():
    if c < MIN_PAIR or single[a] < MIN_TECH or single[b] < MIN_TECH:
        continue
    pab, pa, pb = c / n_doc, single[a] / n_doc, single[b] / n_doc
    npmi = np.log(pab / (pa * pb)) / -np.log(pab) if pab < 1 else 1.0
    rows.append({"기술A": a, "기술B": b, "동시출현": c, "NPMI": round(npmi, 3)})
pairs_df = pd.DataFrame(rows).sort_values("NPMI", ascending=False)
pairs_df.to_csv(OUT / "5_cooccurrence_npmi.csv", index=False, encoding="utf-8-sig")
G = nx.Graph()
for _, r in pairs_df[pairs_df["NPMI"] >= NPMI_MIN].iterrows():
    G.add_edge(r["기술A"], r["기술B"], weight=r["NPMI"])
comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=SEED), key=lambda c: -sum(single[t] for t in c))
say(f"\n[5] 공출현 네트워크: 기술 {G.number_of_nodes()}개, 연결 {G.number_of_edges()}개 (동시출현 ≥ {MIN_PAIR}, NPMI ≥ {NPMI_MIN})")
say("  NPMI 상위 쌍: " + ", ".join(f"{a}+{b}({v:.2f})" for a, b, v in pairs_df.head(10)[["기술A", "기술B", "NPMI"]].values))
crows = []
for i, c in enumerate(comms, 1):
    mem = sorted(c, key=lambda t: -single[t])
    lead = share.loc[mem, GROUPS].astype(float).mean().idxmax()
    crows.append({"묶음": i, "주 직군": lead, "기술 수": len(mem), "구성 기술(빈도순)": ", ".join(mem)})
    say(f"  묶음{i} [{lead}] " + ", ".join(mem[:12]) + (" …" if len(mem) > 12 else ""))
pd.DataFrame(crows).to_csv(OUT / "5_tech_communities.csv", index=False, encoding="utf-8-sig")

# 그림: 노드 색 = 커뮤니티의 주 직군, 크기 = 언급 공고 수. 읽히도록 노드별 가장 강한 연결 3개만 그림
comm_of = {t: i for i, c in enumerate(comms) for t in c}
lead_of = {i: r["주 직군"] for i, r in enumerate(crows)}
H_ = nx.Graph()
H_.add_nodes_from(G.nodes)
for t in G.nodes:
    for u, d in sorted(G[t].items(), key=lambda kv: -kv[1]["weight"])[:3]:
        H_.add_edge(t, u, weight=d["weight"])
pos = nx.spring_layout(H_, weight="weight", seed=7, k=1.0, iterations=300)
fig, ax = plt.subplots(figsize=(12, 10))
w = np.array([d["weight"] for *_, d in H_.edges(data=True)])
nx.draw_networkx_edges(H_, pos, ax=ax, width=0.5 + 3 * w, edge_color="#c3c2b7", alpha=0.7)
nx.draw_networkx_nodes(H_, pos, ax=ax, node_size=[40 + single[t] * 4 for t in H_.nodes],
                       node_color=[COLOR[lead_of[comm_of[t]]] for t in H_.nodes], edgecolors=SURFACE, linewidths=2)
for t, (x, y) in pos.items():
    ax.text(x, y + 0.035, t, ha="center", va="bottom", fontsize=8.5, color=INK)
for g in GROUPS:
    ax.scatter([], [], s=90, color=COLOR[g], label=f"{g} 중심 묶음")
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.set_axis_off()
ax.set_title("기술 공출현 네트워크 (NPMI ≥ 0.1, Louvain 묶음별 색, 노드 크기 = 언급 공고 수)", loc="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig4_cooccurrence_network.png", dpi=160)
plt.close(fig)

# ---------------- 6. 군집화 ----------------
Xd = X.toarray()
sil = {}
for k in range(3, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xd)
    sil[k] = silhouette_score(Xd, km.labels_, metric="cosine", random_state=SEED)
km = KMeans(n_clusters=4, n_init=20, random_state=SEED).fit(Xd)
ward = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(Xd)
df["km"], df["ward"] = km.labels_, ward.labels_
ari_km = adjusted_rand_score(df["label_ko"], df["km"])
ari_w = adjusted_rand_score(df["label_ko"], df["ward"])
say(f"\n[6] 공고 TF-IDF 군집화 (k=4로 직군 수와 맞춤). 실루엣(cosine) k=3~8: " +
    ", ".join(f"{k}:{v:.3f}" for k, v in sil.items()))
say(f"  직군과의 일치도 ARI: K-means {ari_km:.2f}, Ward {ari_w:.2f}")
clu_rows = []
for name, col in [("K-means", "km"), ("Ward", "ward")]:
    ct = pd.crosstab(df[col], df["label_ko"]).reindex(columns=GROUPS, fill_value=0)
    for c in ct.index:
        m = np.asarray(X[(df[col] == c).values].mean(axis=0)).ravel()
        top = [show(terms[j]) for j in np.argsort(m)[::-1][:8]]
        lead = ct.loc[c].idxmax()
        clu_rows.append({"방법": name, "군집": int(c), "공고 수": int(ct.loc[c].sum()), **{g: int(ct.loc[c, g]) for g in GROUPS},
                         "주 직군 비중": f"{lead} {ct.loc[c, lead] / ct.loc[c].sum():.0%}", "대표 단어": ", ".join(top)})
        say(f"  {name} 군집{c} ({int(ct.loc[c].sum())}건, {clu_rows[-1]['주 직군 비중']}): {', '.join(top[:6])}")
clu = pd.DataFrame(clu_rows)
clu.to_csv(OUT / "6_clusters.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(14, 3.9), sharex=True)
for ax, name in zip(axes, ["K-means", "Ward"]):
    s = clu[clu["방법"] == name].sort_values("군집")
    prop = s[GROUPS].div(s["공고 수"], axis=0).values
    y = np.arange(len(s))
    left = np.zeros(len(s))
    for j, g in enumerate(GROUPS):
        ax.barh(y, prop[:, j], left=left, color=COLOR[g], edgecolor=SURFACE, linewidth=2, height=0.66, label=g)
        for i, v in enumerate(prop[:, j]):
            if v >= 0.12:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5, color=INK)
        left += prop[:, j]
    ax.set_yticks(y, [f"군집{c} ({n}건)\n{w.split(', ')[0]}, {w.split(', ')[1]}" for c, n, w in s[["군집", "공고 수", "대표 단어"]].values], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.tick_params(axis="y", length=0)
    ari = ari_km if name == "K-means" else ari_w
    ax.set_title(f"{name} 군집별 직군 구성 (ARI {ari:.2f})", loc="left", color=INK, fontsize=12)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower left", bbox_to_anchor=(0.01, -0.02), frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig5_cluster_composition.png", dpi=160)
plt.close(fig)

# 기술 용어 덴드로그램: 공고 출현 벡터(기술 × 공고)의 Jaccard 거리, average 연결
dend_techs = [t for t, n in single.most_common() if n >= 10][:45]
M = np.array([[t in set(ts) for ts in df["tech_all"]] for t in dend_techs], bool)
Z = linkage(pdist(M, metric="jaccard"), method="average")
cut = fcluster(Z, t=0.85, criterion="distance")
pd.DataFrame({"기술": dend_techs, "언급 공고 수": [single[t] for t in dend_techs], "덴드로그램 묶음(거리 0.85)": cut,
              "가장 많이 언급한 직군": [share.loc[t, GROUPS].astype(float).idxmax() for t in dend_techs]}) \
    .sort_values("덴드로그램 묶음(거리 0.85)").to_csv(OUT / "6_tech_dendrogram_groups.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8, 11.5))
# 가지는 회색으로 통일하고, 색은 기술 이름(가장 많이 언급한 직군)에만 씀
dendrogram(Z, labels=dend_techs, orientation="left", ax=ax, link_color_func=lambda k: "#8f8e88", leaf_font_size=9)
for lbl in ax.get_ymajorticklabels():
    lbl.set_color(COLOR[share.loc[lbl.get_text(), GROUPS].astype(float).idxmax()])
ax.axvline(0.85, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
ax.set_xlabel("Jaccard 거리 (점선 = 묶음 기준 0.85)")
ax.tick_params(axis="y", length=0)
for g in GROUPS:
    ax.plot([], [], "s", color=COLOR[g], label=f"{g}에서 가장 많이 언급")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.05), ncol=2, frameon=False, fontsize=8.5)
ax.set_title("기술 용어 덴드로그램 (함께 요구되는 기술끼리 가까움)", loc="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig6_tech_dendrogram.png", dpi=160)
plt.close(fig)

# ---------------- 7. KWIC ----------------
kw_rows = []
body = df[["duties", "reqs", "prefs"]].fillna("").agg("\n".join, axis=1)
for g in GROUPS:
    for kind, lo in (("기술", lo_t), ("단어", lo_w)):
        for term in lo[g].sort_values(ascending=False).head(5).index:
            sub = df[df["label_ko"] == g]
            n = 0
            for i in sub.index:
                text = body[i].replace("\n", " / ")
                spans = find_spans(text, term) if kind == "기술" else [(m.start(), m.end()) for m in re.finditer(re.escape(term), text, re.I)]
                if spans:
                    a, b = spans[0]
                    kw_rows.append({"직군": g, "종류": kind, "키워드": term, "공고 id": df.at[i, "id"], "출처": df.at[i, "source"],
                                    "왼쪽 맥락": text[max(0, a - 35):a], "키워드 원문": text[a:b], "오른쪽 맥락": text[b:b + 35]})
                    n += 1
                if n >= 3:
                    break
pd.DataFrame(kw_rows).to_csv(OUT / "7_kwic.csv", index=False, encoding="utf-8-sig")
say(f"\n[7] KWIC: 직군별 log-odds 상위 기술·단어 5개씩, 원문 맥락 최대 3건 → 7_kwic.csv ({len(kw_rows)}줄)")

(OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
print(f"\n저장 위치: {OUT}")
