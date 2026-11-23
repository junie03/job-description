"""
최종 기획안 기준 수집기: 점핏 + 사람인 IT 4개 직군 채용공고

점핏 (jumpit.saramin.co.kr)
  - 목록: 웹페이지가 쓰는 공개 목록 API(jumpit-api.saramin.co.kr/api/positions)로 공고 번호 확보
  - 상세: /position/{id} (robots.txt 허용 경로) → 주요업무·자격요건·우대사항 섹션
  - 직군: 서버/백엔드 개발자(1), 프론트엔드 개발자(2), devops/시스템 엔지니어(9), 빅데이터 엔지니어(19)
  - 조건에 맞는 공고 전체 수집
사람인 (www.saramin.co.kr)
  - 목록: /zf_user/jobs/list/job-category, 상세: /zf_user/jobs/relay/view-detail (robots.txt 허용 경로)
  - 직무 태그: 백엔드/서버개발(84), 프론트엔드(92), 데이터엔지니어(83), 인프라(127)
  - 제목(직무명)이 해당 직군과 일치하는 공고를 직군당 50건 목표, 풀스택 공고는 직군당 최대 15건
두 사이트는 같은 공고 번호를 쓰므로, 점핏에 있는 번호는 사람인에서 건너뜀(사이트 간 중복 제거)

사용법: python crawl.py [--saramin-target 50] [--fullstack-max 15] [--delay 1.0]
출력:   data/raw/jumpit.jsonl, data/raw/saramin.jsonl (중단 후 재실행 시 이어받기)
"""
import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 (student text-mining project)",
    "Accept-Language": "ko-KR,ko;q=0.9",
}
RAW = Path("data/raw")

JUMPIT_API = "https://jumpit-api.saramin.co.kr/api/positions"
JUMPIT_VIEW = "https://jumpit.saramin.co.kr/position/{}"
JUMPIT_CATS = {"backend": 1, "frontend": 2, "infra": 9, "data_eng": 19}

SARAMIN_LIST = "https://www.saramin.co.kr/zf_user/jobs/list/job-category"
SARAMIN_DETAIL = "https://www.saramin.co.kr/zf_user/jobs/relay/view-detail"
SARAMIN_CODES = {"backend": 84, "frontend": 92, "data_eng": 83, "infra": 127}

# 사람인 목록 단계에서 '제목이 직군과 일치'하는지 보는 규칙(최종 라벨은 preprocess.py에서 다시 판정)
TITLE_MATCH = {
    "backend":  r"백엔드|백앤드|back[\s-]?end|서버\s*(개발|프로그래머)|서버개발|server\s*(developer|engineer)|java\s*개발|자바\s*개발|spring",
    "frontend": r"프론트|프런트|front[\s-]?end|react|vue|웹\s*프론트",
    "data_eng": r"데이터\s*엔지니어|data\s*engineer|데이터\s*파이프라인|빅데이터\s*(개발|엔지니어)|\betl\b|\bdw\b|데이터\s*플랫폼",
    "infra":    r"인프라|devops|데브옵스|\bsre\b|클라우드\s*(엔지니어|운영|아키텍트)|시스템\s*엔지니어|플랫폼\s*엔지니어|서버\s*운영|시스템\s*운영|infra",
}
FULLSTACK = r"풀스택|full[\s-]?stack"

# 사람인 본문 섹션 제목
SEC = {
    "duties": r"(주요\s*업무|담당\s*업무|업무\s*내용|직무\s*내용|주요\s*직무|하는\s*일|responsibilit|what\s+you('ll)?\s+do)",
    "reqs":   r"(자격\s*요건|지원\s*자격|필수\s*(요건|사항|역량|조건)|자격\s*조건|자격\s*사항|requirement|qualification|이런\s*분을\s*찾)",
    "prefs":  r"(우대\s*사항|우대\s*조건|우대\s*요건|우대\s*역량|prefer|nice\s+to\s+have|이런\s*분이면\s*더)",
    "stop":   r"(근무\s*조건|근무\s*환경|복지|혜택|채용\s*절차|전형\s*절차|접수\s*기간|접수\s*방법|제출\s*서류|유의\s*사항|기타\s*사항|"
              r"고용\s*형태|급여|근무지|근무\s*시간|회사\s*소개|기업\s*소개|모집\s*분야|모집\s*인원|benefit|채용\s*프로세스|문의|지원\s*방법|"
              r"기술\s*스택|tech\s*stack|팀\s*소개)",
}
BLOCK_TAGS = ["p", "div", "li", "tr", "br", "dd", "dt", "h1", "h2", "h3", "h4", "h5", "pre", "table", "ul", "ol"]


def get(url, params=None, as_json=False, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.json() if as_json else r.text
            if r.status_code in (403, 429):
                time.sleep(30 * (i + 1))
        except requests.RequestException:
            time.sleep(5)
    return None


def nap(base):
    time.sleep(base + random.uniform(0, base * 0.5))


def block_text(el):
    """블록 요소 경계에서만 줄을 바꿔 텍스트를 뽑는다.
    인라인 태그(span·b·font 등) 경계에서는 줄을 바꾸지 않으므로 'Typesc/ript'처럼 끊기지 않는다."""
    for br in el.find_all("br"):
        br.replace_with("\n")
    for t in el.find_all(BLOCK_TAGS):
        t.insert_before("\n")
        t.insert_after("\n")
    text = el.get_text("")
    lines = [re.sub(r"[ \t ​]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def load_ids(path):
    ids = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                ids.add(str(json.loads(line)["id"]))
            except Exception:
                pass
    return ids


# ======================= 점핏 =======================
def jumpit_list(delay):
    items = {}
    for g, cat in JUMPIT_CATS.items():
        page = 1
        while True:
            js = get(JUMPIT_API, {"jobCategory": cat, "sort": "reg_dt", "page": page}, as_json=True)
            res = (js or {}).get("result") or {}
            ps = res.get("positions") or []
            if not ps:
                break
            for p in ps:
                rec = items.setdefault(str(p["id"]), {**p, "query_groups": []})
                rec["query_groups"].append(g)
            print(f"[점핏 목록] {g} page {page}: 누적 {sum(g in v['query_groups'] for v in items.values())}/{res.get('totalCount')}")
            if page * len(ps) >= res.get("totalCount", 0) and len(ps) < 16:
                break
            page += 1
            nap(delay * 0.5)
    return items


def jumpit_detail(pid):
    html = get(JUMPIT_VIEW.format(pid))
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    sec = {}
    for dl in soup.find_all("dl"):
        dt, dd = dl.find("dt"), dl.find("dd")
        if dt and dd:
            sec[dt.get_text(strip=True)] = block_text(dd)
    return {"duties": sec.get("주요업무", ""), "reqs": sec.get("자격요건", ""), "prefs": sec.get("우대사항", ""),
            "other_sections": {k: v for k, v in sec.items() if k not in ("주요업무", "자격요건", "우대사항")}}


def crawl_jumpit(delay):
    out = RAW / "jumpit.jsonl"
    done = load_ids(out)
    items = jumpit_list(delay)
    todo = [p for k, p in items.items() if k not in done]
    print(f"[점핏 상세] 목록 {len(items)}건 중 {len(todo)}건 수집")
    with open(out, "a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            d = jumpit_detail(p["id"])
            if d is None:
                continue
            rec = {"source": "jumpit", "id": str(p["id"]), "url": JUMPIT_VIEW.format(p["id"]),
                   "title": p["title"], "company": p["companyName"],
                   "job_tags": p["jobCategory"].split(","), "query_groups": p["query_groups"],
                   "newcomer": p.get("newcomer"), "min_career": p.get("minCareer"), "max_career": p.get("maxCareer"),
                   "career_text": "", "tech_tags": p.get("techStacks", []), "locations": p.get("locations", []),
                   "closed_at": p.get("closedAt"), "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M"), **d}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                f.flush()
                print(f"[점핏 상세] {i}/{len(todo)}")
            nap(delay)
    return set(items)


# ======================= 사람인 =======================
def saramin_parse_list(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for item in soup.select("div.list_item[id^=rec-]"):
        tit = item.select_one(".job_tit a.str_tit")
        comp = item.select_one(".company_nm .str_tit") or item.select_one(".company_nm")
        txt = lambda sel: (item.select_one(sel).get_text(" ", strip=True) if item.select_one(sel) else "")
        rows.append({"id": item["id"].split("-")[1],
                     "title": tit.get("title", tit.get_text(strip=True)) if tit else "",
                     "company": comp.get_text(strip=True) if comp else "",
                     "job_tags": [s.get_text(strip=True) for s in item.select(".job_sector > span")],
                     "career_text": txt(".career"), "location": txt(".work_place")})
    return rows


def heading(line):
    core = re.sub(r"[^\w가-힣\s/&']", " ", line).strip().lower()
    if not core or len(core) > 25:
        return None
    for k in ("prefs", "reqs", "duties", "stop"):
        if re.search(SEC[k], core):
            return k
    return None


def saramin_split(text):
    sec = {"duties": [], "reqs": [], "prefs": []}
    cur = None
    for ln in text.split("\n"):
        h = heading(ln)
        if h:
            cur = None if h == "stop" else h
            rest = re.split(r"[:：]", ln, maxsplit=1)
            if cur and len(rest) == 2 and rest[1].strip():
                sec[cur].append(rest[1].strip())
            continue
        if cur:
            sec[cur].append(ln)
    return {k: "\n".join(v) for k, v in sec.items()}


def saramin_detail(rid):
    html = get(SARAMIN_DETAIL, {"rec_idx": rid, "rec_seq": 0})
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    body = soup.select_one(".user_content") or soup.body or soup
    n_img = sum(1 for i in body.find_all("img") if "static/hiring/images/template" not in (i.get("src") or ""))
    text = block_text(body)
    return {**saramin_split(text), "full_text": text, "n_content_imgs": n_img}


def saramin_career(text):
    """'신입', '경력 3년↑', '3 ~ 10년', '신입 · 경력', '경력무관' → (min, max, newcomer)"""
    t = text or ""
    m = re.search(r"(\d+)\s*~\s*(\d+)\s*년", t)
    if m:
        return int(m.group(1)), int(m.group(2)), "신입" in t
    m = re.search(r"경력\s*(\d+)\s*년", t)
    if m:
        return int(m.group(1)), None, "신입" in t
    if "신입" in t or "경력무관" in t:
        return 0, None, True
    return None, None, None


def crawl_saramin(target, fullstack_max, delay, jumpit_ids):
    out = RAW / "saramin.jsonl"
    done = load_ids(out)
    picked = {}
    for g, code in SARAMIN_CODES.items():
        n_main = n_full = 0
        page = 1
        while (n_main < target or n_full < fullstack_max) and page <= 20:
            html = get(SARAMIN_LIST, {"cat_kewd": code, "page": page, "page_count": 100, "sort": "RD"})
            rows = saramin_parse_list(html) if html else []
            if not rows:
                break
            for r in rows:
                if r["id"] in jumpit_ids or r["id"] in picked:
                    continue
                t = r["title"].lower()
                if re.search(FULLSTACK, t, re.I):
                    if n_full < fullstack_max:
                        picked[r["id"]] = {**r, "query_groups": [g], "pick": "fullstack"}
                        n_full += 1
                elif re.search(TITLE_MATCH[g], t, re.I) and n_main < target:
                    picked[r["id"]] = {**r, "query_groups": [g], "pick": "title_match"}
                    n_main += 1
            print(f"[사람인 목록] {g} page {page}: 제목 일치 {n_main}/{target}, 풀스택 {n_full}/{fullstack_max}")
            page += 1
            nap(delay)
    todo = [r for k, r in picked.items() if k not in done]
    print(f"[사람인 상세] 선택 {len(picked)}건 중 {len(todo)}건 수집")
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            d = saramin_detail(r["id"])
            if d is None:
                continue
            mn, mx, newc = saramin_career(r["career_text"])
            rec = {"source": "saramin", "id": r["id"],
                   "url": f"https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx={r['id']}",
                   "title": r["title"], "company": r["company"], "job_tags": r["job_tags"],
                   "query_groups": r["query_groups"], "pick": r["pick"],
                   "newcomer": newc, "min_career": mn, "max_career": mx, "career_text": r["career_text"],
                   "tech_tags": [], "locations": [r["location"]],
                   "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M"), **d}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                f.flush()
                print(f"[사람인 상세] {i}/{len(todo)}")
            nap(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saramin-target", type=int, default=50)
    ap.add_argument("--fullstack-max", type=int, default=15)
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    jumpit_ids = crawl_jumpit(a.delay)
    crawl_saramin(a.saramin_target, a.fullstack_max, a.delay, jumpit_ids)
    print("완료")


if __name__ == "__main__":
    main()
