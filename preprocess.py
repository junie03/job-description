"""
최종 기획안 전처리: data/raw/{jumpit,saramin}.jsonl → data/processed/postings.csv

1. 본문 점검: 주요업무·자격요건·우대사항을 찾지 못했거나 본문이 이미지뿐인 공고 제외
2. 직군 판정
   - 기업이 단 직무 태그를 라벨 후보로 쓰고, 공고 제목(직무명)이 그 직군과 맞는지 규칙으로 검토
   - 다른 직무(AI 연구·HW·로봇·보안·QA·영업 등)는 제외
   - 대상 직군 태그가 여러 개면 제목으로 한 직군에 배정(복구)
   - 풀스택 공고는 '경계 공고'로 표시
   - 규칙으로 판정하기 어려운 공고는 data/manual/manual_review.csv로 내보내고,
     data/manual/manual_labels.csv(원문을 읽고 내린 판정 + 근거)가 있으면 그 판정을 적용
3. 텍스트 정제: URL·이메일·전화번호·회사명·특수문자 제거, '~ 우대'로 끝나는 줄은 우대사항으로 이동
4. 기술 용어 추출(tech_dict.py) + Kiwi 형태소 분석(일반명사·고유명사·영문, IT 복합명사 사용자 사전, 불용어)
5. 유사 공고 그룹: 같은 회사의 본문 TF-IDF 코사인 유사도 0.9 이상 → 같은 그룹(분석엔 그룹당 1건, 분류 검증엔 같은 폴드)
6. 경력 구간: 최소 요구 연차 0~2년 신입·주니어 / 3~5년 미들 / 6년 이상 시니어
7. 사전 점검: 점핏 기술스택 태그 중 사전이 못 잡는 태그 → data/processed/dict_check.csv
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tech_dict import TECH, TOKEN, extract

RAW, OUT, MAN = Path("data/raw"), Path("data/processed"), Path("data/manual")
GROUPS = ["backend", "frontend", "infra", "data_eng"]
GROUP_KO = {"backend": "백엔드/서버개발", "frontend": "프론트엔드", "infra": "인프라", "data_eng": "데이터엔지니어"}

TAG2GROUP = {
    "서버/백엔드 개발자": "backend", "프론트엔드 개발자": "frontend",
    "devops/시스템 엔지니어": "infra", "빅데이터 엔지니어": "data_eng",       # 점핏
    "백엔드/서버개발": "backend", "프론트엔드": "frontend", "인프라": "infra", "데이터엔지니어": "data_eng",  # 사람인
}
TITLE = {
    "backend":  r"백엔드|백앤드|back[\s-]?end|서버\s*(개발|프로그래머|엔지니어)|서버개발|\bserver\b|\bapi\s*개발|java\s*개발|자바\s*개발|spring|\bbe\b|플랫폼\s*개발",
    "frontend": r"프론트|프런트|프론드|front[\s-]?end|\bfe\b|react|vue|웹\s*프론트|\bui\s*개발",
    "infra":    r"인프라|infra|devops|데브옵스|\bsre\b|클라우드|cloud|시스템\s*엔지니어|시스템\s*운영|서버\s*운영|플랫폼\s*엔지니어|platform\s*engineer|네트워크\s*엔지니어|kubernetes|쿠버네티스",
    "data_eng": r"데이터\s*엔지니어|data\s*engineer|데이터\s*파이프라인|빅데이터|big\s*data|\betl\b|\bdw\b|데이터\s*플랫폼|data\s*platform|데이터\s*개발",
}
FULLSTACK = r"풀스택|풀스텍|full[\s-]?stack"
# 제목이 백엔드·프론트엔드를 함께 가리키는 웹 개발 공고도 풀스택 경계로 봄
WEB_BOTH = r"웹\s*(서비스\s*)?개발자|웹\s*어플리케이션\s*개발|웹\s*/\s*앱|product\s*engineer"
# 대상 4개 직군이 아닌 직무(제목에만 이 단어가 있고 대상 직군 단어가 없으면 제외)
OTHER_JOB = (r"ai\s*(연구|research|모델|엔지니어)|머신\s*러닝|machine\s*learning|\bml\s*엔지니어|딥\s*러닝|llm\s*엔지니어|연구원|researcher|"
             r"\bhw\b|하드웨어|hardware|임베디드|embedded|펌웨어|firmware|로봇|robot|회로|반도체|fpga|plc|"
             r"정보\s*보안|보안\s*(관제|컨설|엔지니어|담당)|security|\bqa\b|테스트\s*엔지니어|tester|"
             r"영업|세일즈|sales|기술\s*지원|pm\b|pmo|기획|디자이너|designer|퍼블리셔|마케팅|"
             r"ios|android|안드로이드|앱\s*개발|모바일\s*앱|게임\s*클라이언트|unity|언리얼|dba\b|강사|교육생|"
             r"데이터\s*(분석|사이언티스트)|data\s*(analyst|scientist)|erp|sap\b|mes\b|"
             # 점핏 태그가 느슨해 대상 직군 태그가 붙은 비대상 직무(원자료 검토로 추가)
             r"게임|광학|기구\s*설계|공정|제조|안테나|antenna|\brf\b|mmic|physical\s*design|자율\s*주행|slam|perception|"
             r"비전|vision|측위|시뮬레이션|제어|설비|전장|영상\s*처리|llm|deep\s*learning|ai\s*(sw|개발|서비스|애플리케이션|artist|engineer)|"
             r"ai/ml|\bax\b|정보\s*보호|개인\s*정보|보안|관제|컨택\s*센터|\bcti\b|\bpbx\b|microsoft\s*365|온톨로지|ontology|지도\s*데이터|"
             r"specialist|manager|매니저|architect|아키텍트|\bta\b|\baa\b|\bfde\b|field|필드|검증|v2x|eai|tibco|hw|pcb|mapf|"
             r"iot|로보어드바이저|irp|daas|adc|fms")

STOPWORDS = set("""
우대 우대사항 우대함 경험 경험자 가능 가능자 이상 이하 보유 보유자 관련 분야 업무 담당 수행 능력 역량 지식 이해 활용 사용 기반
필수 자격 요건 사항 조건 주요 지원 지원자 학력 무관 대졸 초대졸 졸업 예정 예정자 년 년차 경력 신입 개월 명 등 및 분 자 것 수 중
해당 각종 다양 전반 기타 내용 경우 정도 수준 부분 위주 대한 위한 통한 통해 함께 성장 회사 당사 팀 부서 근무 채용 모집 포지션 직무
합류 입사 지원서 이력서 포트 서류 전형 면접 협의 결정 연봉 급여 복지 혜택 근무지 위치 시간 주 일 월 개 곳 외 내 후 전
""".split())
USER_WORDS = ["파이프라인", "온프레미스", "마이크로서비스", "데이터레이크", "데이터웨어하우스", "레이크하우스", "트러블슈팅", "코드리뷰",
              "대용량", "고가용성", "무중단", "리팩토링", "모니터링", "자동화", "컨테이너", "오케스트레이션", "클라우드네이티브", "스트리밍",
              "프레임워크", "아키텍처", "인프라", "백엔드", "프론트엔드", "풀스택", "데브옵스", "웹소켓", "캐싱", "샤딩", "레플리케이션"]

URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\d{2,4}[-.\s]\d{3,4}[-.\s]\d{4}")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
BULLET = re.compile(r"^[\s\-–—•·ㆍ∙▶▷►※*○●◦■□◆◇✓✔☞→>\d.)]+")
PREF_TAIL = re.compile(r"(우대|우대함|우대합니다|우대해요|가산점)\s*[.!)]*\s*$")


def clean(s, company=""):
    if not isinstance(s, str) or not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for rx in (URL, EMAIL, PHONE, EMOJI):
        s = rx.sub(" ", s)
    if company:
        core = re.sub(r"\(주\)|㈜|주식회사|\(유\)|\s", "", company)
        if len(core) >= 2:
            s = re.sub(re.escape(core), " ", s)
    out = []
    for ln in s.split("\n"):
        ln = BULLET.sub("", ln)
        ln = re.sub(r"[^\w가-힣\s.+#/\-]", " ", ln)       # 특수문자 제거(기술 표기에 쓰는 . + # / - 는 유지)
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) >= 2:
            out.append(ln)
    return "\n".join(out)


def group_of_tags(tags):
    return sorted({TAG2GROUP[t.strip()] for t in tags if t.strip() in TAG2GROUP})


def title_hits(title):
    t = (title or "").lower()
    return [g for g, rx in TITLE.items() if re.search(rx, t, re.I)]


def judge(title, tags):
    """반환: (라벨 또는 None, 판정 방법, 상태) 상태 ∈ keep / boundary / exclude / manual"""
    hits = title_hits(title)
    t = (title or "").lower()
    # 영업·마케팅·강사 공고는 제목에 직군 단어(예: 'IT인프라 영업')가 있어도 제외
    if re.search(r"영업|세일즈|sales|마케팅|강사", t, re.I):
        return None, "제목: 영업·마케팅·강사", "exclude"
    if re.search(FULLSTACK, t, re.I):
        return None, "제목: 풀스택", "boundary"
    if {"backend", "frontend"} <= set(hits) or re.search(WEB_BOTH, t, re.I):
        return None, "제목: 백엔드+프론트엔드(웹 개발)", "boundary"
    if not hits and re.search(OTHER_JOB, t, re.I):
        return None, "제목: 다른 직무", "exclude"
    if len(hits) == 1 and hits[0] not in tags:
        # 제목에 다른 직무 단어나 '운영/개발'·전산 같은 혼합 표현이 같이 있으면 규칙으로 정하지 않음
        if re.search(OTHER_JOB, t, re.I) or re.search(r"운영\s*(/|및)\s*개발|sap|총무|전산|erp", t, re.I):
            return None, "제목에 직군 단어와 다른 직무 단어가 함께 있음", "manual"
        return hits[0], "제목 직무명 우선(태그와 다름)", "keep"
    if len(tags) == 1:
        g = tags[0]
        if g in hits and len(hits) == 1:
            return g, "태그+제목 일치", "keep"
        if g in hits:
            return g, "태그+제목 일치(제목에 다른 직군 단어도 있음)", "keep"
        return None, "태그와 제목 불일치/제목에 직군 단어 없음", "manual"
    cand = [g for g in hits if g in tags]
    if len(cand) == 1:
        return cand[0], "다중 태그 → 제목으로 배정", "keep"
    return None, "다중 태그, 제목으로 결정 불가", "manual"


def exp_band(newcomer, mn):
    if mn is None or (isinstance(mn, float) and np.isnan(mn)):
        return "신입·주니어" if newcomer else "미상"
    mn = int(mn)
    if mn <= 2:
        return "신입·주니어"
    if mn <= 5:
        return "미들"
    return "시니어"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in ("jumpit", "saramin"):
        p = RAW / f"{src}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates("id", keep="first")
    log = [f"수집: {len(df)}건 (점핏 {(df.source == 'jumpit').sum()}, 사람인 {(df.source == 'saramin').sum()})"]

    def step(mask, why):
        nonlocal df
        n = int((~mask).sum())
        df = df[mask].copy()
        log.append(f"- {why}: {n}건 제외 → {len(df)}건")

    # 1. 본문 점검
    sec_len = df[["duties", "reqs", "prefs"]].fillna("").apply(lambda r: sum(len(x) for x in r), axis=1)
    has_two = df[["duties", "reqs", "prefs"]].fillna("").apply(lambda r: sum(len(x) >= 10 for x in r), axis=1) >= 2
    step((sec_len >= 80) & has_two, "본문이 이미지뿐이거나 섹션(주요업무·자격요건·우대사항)을 찾지 못함")
    # 헤드헌팅 공고 중 'JD 비공개' 안내문만 있는 공고: 본문이 제목의 직무와 무관해 직무 내용을 알 수 없음(분류 오분류 검토에서 발견)
    hidden = df["duties"].fillna("").str.contains(r"JD\s*를?\s*비공개|세부\s*채용\s*정보\s*및\s*JD", regex=True)
    step(~hidden, "헤드헌팅 'JD 비공개' 안내문만 있는 공고")

    # 2. 직군 판정
    df["tag_groups"] = df["job_tags"].map(group_of_tags)
    res = df.apply(lambda r: judge(r["title"], r["tag_groups"]), axis=1)
    df["label"], df["label_method"], df["status"] = zip(*res)
    log.append("  판정 결과: " + str(df["status"].value_counts().to_dict()))
    log.append("  다중 태그 → 제목으로 배정(복구): " + str(int((df["label_method"] == "다중 태그 → 제목으로 배정").sum())) + "건")

    review = df[df["status"] == "manual"]
    review[["id", "source", "url", "company", "title", "job_tags", "tag_groups", "duties", "reqs"]] \
        .to_csv(MAN / "manual_review.csv", index=False, encoding="utf-8-sig")
    mpath = MAN / "manual_labels.csv"
    if mpath.exists():
        man = pd.read_csv(mpath, dtype={"id": str}).set_index("id")
        for i, r in df[df["status"] == "manual"].iterrows():
            if r["id"] in man.index:
                d = str(man.loc[r["id"], "decision"]).strip()
                df.at[i, "label_method"] = "원문 판정: " + str(man.loc[r["id"], "reason"])
                if d in GROUPS:
                    df.at[i, "label"], df.at[i, "status"] = d, "keep"
                elif d == "boundary":
                    df.at[i, "status"] = "boundary"
                else:
                    df.at[i, "status"] = "exclude"
        log.append(f"  원문 판정 적용: {int(df['label_method'].str.startswith('원문 판정').sum())}건 (manual_labels.csv)")
    n_left = int((df["status"] == "manual").sum())
    if n_left:
        log.append(f"  ※ 원문 판정이 아직 없는 공고 {n_left}건은 제외됨 → data/manual/manual_review.csv 확인")
    step(df["status"].isin(["keep", "boundary"]), "다른 직무/판정 불가 제외")
    df["label_ko"] = df["label"].map(GROUP_KO).fillna("경계(풀스택)")
    df["is_boundary"] = df["status"] == "boundary"

    # 3. 정제
    for c in ("duties", "reqs", "prefs"):
        df[c] = df.apply(lambda r: clean(r[c], r["company"]), axis=1)

    def move(r):
        prefs = r["prefs"].split("\n") if r["prefs"] else []
        for c in ("reqs", "duties"):
            keep = []
            for ln in (r[c].split("\n") if r[c] else []):
                (prefs if PREF_TAIL.search(ln) else keep).append(ln)
            r[c] = "\n".join(keep)
        r["prefs"] = "\n".join(prefs)
        return r
    df = df.apply(move, axis=1)

    # 4. 기술·토큰
    for c in ("duties", "reqs", "prefs"):
        df[f"tech_{c}"] = df[c].map(extract)
    df["tech_all"] = df.apply(lambda r: sorted(set(r["tech_duties"]) | set(r["tech_reqs"]) | set(r["tech_prefs"])), axis=1)

    from kiwipiepy import Kiwi
    from tech_dict import COMPILED
    kiwi = Kiwi()
    for w in USER_WORDS:
        kiwi.add_user_word(w, "NNG")

    def tokens(text):
        if not text:
            return []
        t = text.lower()
        techs = []
        for c, _, rx in sorted(COMPILED, key=lambda x: -max(len(p) for p in TECH[x[0]][1])):
            n = len(rx.findall(t))
            if n:
                techs += [TOKEN[c]] * n
                t = rx.sub(" ", t)
        words = [k.form.lower() for k in kiwi.tokenize(t) if k.tag in ("NNG", "NNP", "SL")]
        return techs + [w for w in words if len(w) > 1 and w not in STOPWORDS]

    for c in ("duties", "reqs", "prefs"):
        df[f"tok_{c}"] = df[c].map(lambda s: " ".join(tokens(s)))
    df["tokens"] = df[["tok_duties", "tok_reqs", "tok_prefs"]].agg(" ".join, axis=1).str.strip()

    # 5. 유사 공고 그룹
    body = df[["duties", "reqs", "prefs"]].agg("\n".join, axis=1)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1).fit_transform(body)
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    comp_key = df["company"].str.replace(r"\(주\)|㈜|주식회사|\s", "", regex=True).values
    for comp in set(comp_key):
        idx = np.where(comp_key == comp)[0]
        if len(idx) < 2:
            continue
        S = cosine_similarity(X[idx])
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if S[a, b] >= 0.9:
                    parent[find(idx[a])] = find(idx[b])
    roots = [find(i) for i in range(len(df))]
    df["sim_group"] = pd.factorize(pd.Series(roots))[0]
    df["is_rep"] = ~pd.Series(roots).duplicated().values      # 그룹 대표(처음 나온 공고)
    log.append(f"- 유사 공고 그룹: {df['sim_group'].nunique()}개 그룹 (그룹에 묶여 분석에서 빠지는 공고 {int((~df['is_rep']).sum())}건)")

    # 6. 경력 구간
    df["exp_band"] = df.apply(lambda r: exp_band(r["newcomer"], r["min_career"]), axis=1)

    # 7. 사전 점검(점핏 태그)
    tags = Counter(t for ts in df.loc[df["source"] == "jumpit", "tech_tags"] for t in ts)
    chk = [{"jumpit_tag": t, "count": n, "matched": ", ".join(extract(t)) or ""} for t, n in tags.most_common()]
    pd.DataFrame(chk).to_csv(OUT / "dict_check.csv", index=False, encoding="utf-8-sig")
    miss = [c for c in chk if not c["matched"]]
    log.append(f"- 사전 점검: 점핏 기술스택 태그 {len(chk)}종 중 사전이 못 잡은 태그 {len(miss)}종 "
               f"(상위: {', '.join(c['jumpit_tag'] for c in miss[:10])})")

    # 저장
    for c in [c for c in df.columns if c.startswith("tech_")] + ["job_tags", "tag_groups", "locations"]:
        df[c] = df[c].map(lambda v: "|".join(v) if isinstance(v, list) else v)
    cols = ["id", "source", "url", "company", "title", "job_tags", "tag_groups", "label", "label_ko", "label_method",
            "is_boundary", "newcomer", "min_career", "max_career", "career_text", "exp_band", "sim_group", "is_rep",
            "duties", "reqs", "prefs", "tech_duties", "tech_reqs", "tech_prefs", "tech_all", "tech_tags",
            "tok_duties", "tok_reqs", "tok_prefs", "tokens", "collected_at"]
    df[cols].to_csv(OUT / "postings.csv", index=False, encoding="utf-8-sig")

    core = df[~df["is_boundary"]]
    log += ["", f"최종: {len(df)}건 (직군 {len(core)}건 + 풀스택 경계 {int(df['is_boundary'].sum())}건)",
            "직군별: " + str(core["label_ko"].value_counts().to_dict()),
            "출처별: " + str(df["source"].value_counts().to_dict()),
            "직군 × 출처:\n" + pd.crosstab(core["label_ko"], core["source"]).to_string(),
            "직군 × 경력 구간:\n" + pd.crosstab(core["label_ko"], core["exp_band"]).to_string()]
    rep = "\n".join(log)
    (OUT / "preprocess_report.txt").write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
