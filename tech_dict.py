"""
기술 용어 사전(최종 기획안): 표준어 → (분류, [표기 변형 정규식])
- 한글/영문/약어 표기 변형을 하나의 표준어로 통합 (자바→Java, 쿠버네티스·k8s→Kubernetes, 스프링 부트→Spring Boot)
- 복합 용어(Spring Boot, React Native, GitHub Actions 등)는 하나의 토큰으로 보존
- 영문 변형은 앞뒤가 영문·숫자가 아닐 때만 매칭(java ≠ javascript)
- 점핏 기술스택 태그와의 대조·누락 점검은 preprocess.py에서 수행(결과: data/processed/dict_check.csv)
"""
import re

T = {}


def add(cat, items):
    for canon, pats in items.items():
        T[canon] = (cat, pats)


add("language", {
    "Java": [r"java(?!\s*script)", r"자바(?!\s*스크립트)"],
    "JavaScript": [r"javascript", r"자바\s*스크립트", r"js(?!on|p|x)", r"es6", r"ecmascript"],
    "TypeScript": [r"typescript", r"타입\s*스크립트", r"ts(?!x)"],
    "Python": [r"python", r"파이썬"],
    "Kotlin": [r"kotlin", r"코틀린"],
    "Go": [r"golang", r"go\s*언어", r"go(?=\s*(,|/|\)|lang|개발|기반|경험|언어|$))"],
    "C": [r"c(?=\s*(언어|/c\+\+|,\s*c\+\+))", r"c언어"],
    "C++": [r"c\+\+", r"cpp"],
    "C#": [r"c#", r"c\s*sharp"],
    "PHP": [r"php"],
    "Ruby": [r"ruby(?!\s*on)", r"루비"],
    "Scala": [r"scala", r"스칼라"],
    "Rust": [r"rust"],
    "Swift": [r"swift"],
    "Dart": [r"dart"],
    "R": [r"r(?=\s*(언어|,\s*python|/python))", r"r언어"],
    "SQL": [r"sql(?!\s*server)", r"쿼리\s*(작성|튜닝|최적화)"],
    "Shell Script": [r"shell\s*script", r"bash", r"쉘\s*스크립트", r"셸\s*스크립트", r"shell"],
    "PowerShell": [r"powershell"],
    "HTML": [r"html5?"],
    "CSS": [r"css3?"],
    "Sass/SCSS": [r"scss", r"sass"],
    "Groovy": [r"groovy"],
    "Elixir": [r"elixir"],
    "Lua": [r"lua"],
    "Perl": [r"perl"],
    "Solidity": [r"solidity"],
})
add("backend", {
    "Spring Boot": [r"spring\s*boot", r"스프링\s*부트", r"springboot"],
    "Spring": [r"spring(?!\s*(boot|batch|cloud|security|webflux|data))", r"스프링(?!\s*부트)"],
    "Spring Batch": [r"spring\s*batch", r"스프링\s*배치"],
    "Spring Cloud": [r"spring\s*cloud"],
    "Spring Security": [r"spring\s*security"],
    "WebFlux": [r"webflux", r"spring\s*webflux"],
    "JPA": [r"jpa", r"hibernate", r"하이버네이트"],
    "QueryDSL": [r"querydsl"],
    "MyBatis": [r"mybatis", r"ibatis", r"마이바티스"],
    "JSP": [r"jsp"],
    "Node.js": [r"node\.?js", r"노드\s*js", r"nodejs"],
    "Express": [r"express(\.?js)?"],
    "NestJS": [r"nest\.?js"],
    "Django": [r"django", r"장고"],
    "FastAPI": [r"fast\s*api"],
    "Flask": [r"flask"],
    "Ruby on Rails": [r"ruby\s*on\s*rails", r"rails"],
    "Laravel": [r"laravel"],
    ".NET": [r"\.net(?!\w)", r"asp\.net", r"닷넷", r"dotnet"],
    "gRPC": [r"grpc"],
    "GraphQL": [r"graphql"],
    "REST API": [r"rest(ful)?\s*api", r"restful", r"rest"],
    "WebSocket": [r"websocket", r"웹\s*소켓", r"socket\.io"],
    "MSA": [r"msa", r"마이크로\s*서비스", r"micro\s*services?"],
    "DDD": [r"ddd", r"도메인\s*주도\s*설계"],
    "TDD": [r"tdd", r"테스트\s*주도\s*개발"],
    "OAuth": [r"oauth\s*2?(\.0)?", r"oidc"],
    "JWT": [r"jwt"],
    "Netty": [r"netty"],
    "Tomcat": [r"tomcat", r"톰캣"],
    "WAS": [r"was", r"jeus", r"weblogic", r"웹\s*애플리케이션\s*서버"],
    "eGovFrame": [r"전자정부\s*(표준)?\s*프레임워크", r"egov"],
    "JUnit": [r"junit"],
    "pytest": [r"pytest"],
    "Celery": [r"celery"],
    "MQTT": [r"mqtt"],
    "Gradle": [r"gradle"],
    "Maven": [r"maven"],
})
add("frontend", {
    "React": [r"react(\.js)?(?!\s*(native|query))", r"리액트(?!\s*네이티브)"],
    "React Native": [r"react\s*native", r"리액트\s*네이티브"],
    "Vue.js": [r"vue(\.js)?(?!x)", r"뷰\s*js"],
    "Nuxt.js": [r"nuxt(\.js)?"],
    "Angular": [r"angular(js)?"],
    "Next.js": [r"next\.?js"],
    "Svelte": [r"svelte(kit)?"],
    "jQuery": [r"jquery", r"제이쿼리"],
    "Redux": [r"redux(\s*toolkit)?", r"rtk"],
    "Recoil": [r"recoil"],
    "Zustand": [r"zustand"],
    "MobX": [r"mobx"],
    "Jotai": [r"jotai"],
    "Vuex/Pinia": [r"vuex", r"pinia"],
    "React Query": [r"react[\s-]*query", r"tanstack\s*query"],
    "SWR": [r"swr"],
    "Webpack": [r"webpack", r"웹팩"],
    "Vite": [r"vite"],
    "Babel": [r"babel"],
    "Storybook": [r"storybook", r"스토리북"],
    "Tailwind CSS": [r"tailwind(\s*css)?"],
    "Styled Components": [r"styled[\s-]*components?"],
    "Emotion": [r"emotion"],
    "Bootstrap": [r"bootstrap"],
    "MUI": [r"\bmui\b", r"material[\s-]*ui"],
    "Three.js": [r"three\.?js"],
    "D3.js": [r"d3(\.js)?"],
    "Chart.js": [r"chart\.?js", r"echarts", r"highcharts"],
    "PWA": [r"pwa"],
    "SSR/CSR": [r"ssr", r"csr", r"서버\s*사이드\s*렌더링"],
    # '성능 최적화'만으로는 서버·DB·인프라 성능도 가리키므로 웹·렌더링 맥락의 표현만 매핑(KWIC 점검으로 오매핑 확인)
    "웹 성능 최적화": [r"(웹|프론트엔드|프론트|렌더링|로딩|페이지)\s*(성능|속도)", r"web\s*vitals", r"lighthouse", r"번들\s*(크기|사이즈)", r"lcp", r"cls"],
    "웹 표준·접근성": [r"웹\s*표준", r"웹\s*접근성", r"크로스\s*브라우징", r"wcag"],
    "반응형 웹": [r"반응형"],
    "웹 퍼블리싱": [r"퍼블리싱"],
    "Figma": [r"figma", r"피그마"],
    "Jest": [r"jest"],
    "Axios": [r"axios"],
    "RxJS": [r"rxjs"],
    "WebRTC": [r"webrtc"],
    "Selenium": [r"selenium", r"셀레니움"],
    "Cypress": [r"cypress"],
    "Playwright": [r"playwright"],
    "ESLint/Prettier": [r"eslint", r"prettier"],
    "Monorepo": [r"monorepo", r"모노레포", r"turborepo", r"nx"],
    "Flutter": [r"flutter", r"플러터"],
    "Electron": [r"electron"],
})
add("database", {
    "MySQL": [r"mysql"],
    "MariaDB": [r"maria\s*db", r"마리아\s*db"],
    "PostgreSQL": [r"postgre(sql|s)?", r"포스트그레"],
    "Oracle": [r"oracle(\s*db)?", r"오라클"],
    "MSSQL": [r"ms[\s-]*sql", r"sql\s*server"],
    "Tibero": [r"tibero", r"티베로"],
    "MongoDB": [r"mongo(db)?", r"몽고\s*db"],
    "Redis": [r"redis", r"레디스"],
    "Elasticsearch": [r"elastic\s*search", r"엘라스틱\s*서치", r"opensearch"],
    "DynamoDB": [r"dynamo\s*db"],
    "Cassandra": [r"cassandra"],
    "HBase": [r"hbase"],
    "Neo4j": [r"neo4j"],
    "SQLite": [r"sqlite"],
    "Memcached": [r"memcached"],
    "ClickHouse": [r"clickhouse"],
    "Vector DB": [r"vector\s*db", r"벡터\s*db", r"벡터\s*데이터베이스", r"milvus", r"pinecone", r"qdrant", r"weaviate", r"pgvector", r"chroma"],
    "RDBMS": [r"rdbms", r"rdb", r"관계형\s*데이터베이스"],
    "NoSQL": [r"nosql"],
    "DB 설계": [r"(db|데이터베이스)\s*(설계|모델링)", r"데이터\s*모델링", r"erd"],
    "쿼리 튜닝": [r"쿼리\s*(튜닝|최적화)", r"sql\s*튜닝", r"인덱스\s*설계"],
})
add("cloud", {
    "AWS": [r"aws", r"amazon\s*web\s*services?", r"아마존\s*웹"],
    "AWS EC2": [r"ec2"],
    "AWS S3": [r"s3"],
    "AWS Lambda": [r"lambda", r"람다"],
    "AWS RDS": [r"rds", r"aurora"],
    "AWS ECS": [r"ecs", r"fargate"],
    "AWS EKS": [r"eks"],
    "CloudFormation": [r"cloudformation"],
    "GCP": [r"gcp", r"google\s*cloud(\s*platform)?", r"구글\s*클라우드"],
    "Azure": [r"azure", r"애저", r"애져"],
    "NCP": [r"ncp", r"naver\s*cloud", r"네이버\s*클라우드"],
    "KT Cloud": [r"kt\s*cloud", r"kt\s*클라우드"],
    "Firebase": [r"firebase", r"파이어베이스"],
    "Serverless": [r"serverless", r"서버리스"],
    "CDN": [r"cdn", r"cloudfront", r"cloudflare"],
    "멀티·하이브리드 클라우드": [r"멀티\s*클라우드", r"하이브리드\s*클라우드", r"multi[\s-]*cloud", r"hybrid\s*cloud"],
    "온프레미스": [r"온\s*프레미스", r"on[\s-]*premise"],
})
add("devops", {
    "Linux": [r"linux", r"리눅스", r"ubuntu", r"centos", r"rhel", r"rocky\s*linux", r"우분투"],
    "Unix": [r"unix", r"유닉스", r"aix", r"solaris", r"hp-?ux"],
    "Windows": [r"windows(?!\s*server)", r"윈도우(?!\s*서버)"],
    "Windows Server": [r"windows\s*server", r"윈도우\s*서버", r"active\s*directory"],
    "Docker": [r"docker", r"도커"],
    "Kubernetes": [r"kubernetes", r"k8s", r"쿠버네티스", r"쿠버"],
    "Helm": [r"helm"],
    "Istio": [r"istio", r"service\s*mesh", r"서비스\s*메시"],
    "Terraform": [r"terraform", r"테라폼"],
    "Ansible": [r"ansible", r"앤서블"],
    "IaC": [r"iac", r"infrastructure\s*as\s*code"],
    "Jenkins": [r"jenkins", r"젠킨스"],
    "GitHub Actions": [r"github\s*actions?"],
    "GitLab CI": [r"gitlab\s*ci"],
    "ArgoCD": [r"argo\s*cd", r"argocd", r"argo"],
    "CI/CD": [r"ci\s*/\s*cd", r"ci\s*cd", r"지속적\s*(통합|배포)"],
    "Nginx": [r"nginx", r"엔진엑스"],
    "Apache HTTP": [r"apache(?!\s*(spark|kafka|airflow|flink|hadoop|hive|iceberg|beam|nifi))", r"httpd"],
    "Prometheus": [r"prometheus", r"프로메테우스"],
    "CloudWatch": [r"cloud\s*watch"],
    "API Gateway": [r"api\s*gateway", r"kong", r"api\s*게이트웨이"],
    "Grafana": [r"grafana", r"그라파나"],
    "ELK": [r"elk", r"logstash", r"kibana", r"fluentd", r"fluent\s*bit", r"loki"],
    "Datadog": [r"datadog"],
    "New Relic/APM": [r"new\s*relic", r"\bapm\b", r"scouter", r"pinpoint", r"whatap", r"제니퍼"],
    "모니터링": [r"모니터링", r"monitoring", r"observability", r"관측성"],
    "VMware": [r"vmware", r"vsphere", r"esxi", r"vcenter"],
    "가상화": [r"가상화", r"hypervisor", r"하이퍼바이저", r"kvm"],
    "OpenStack": [r"openstack", r"오픈스택"],
    "네트워크": [r"네트워크", r"network", r"tcp/ip", r"l2", r"l3", r"l4", r"라우팅", r"스위치", r"vlan", r"vpn", r"cisco", r"l7", r"ethernet"],
    "방화벽": [r"방화벽", r"firewall", r"waf", r"ips", r"ids"],
    "DNS": [r"dns"],
    "로드밸런서": [r"로드\s*밸런서", r"load\s*balanc", r"\belb\b", r"\balb\b", r"haproxy"],
    "스토리지": [r"스토리지", r"storage", r"nas", r"san", r"ceph"],
    "백업·DR": [r"백업", r"backup", r"재해\s*복구", r"\bdr\b"],
    "보안": [r"보안", r"security", r"isms", r"취약점"],
    "SRE": [r"sre", r"site\s*reliability"],
    "DevOps": [r"devops", r"데브옵스"],
    "GitOps": [r"gitops"],
    "MLOps": [r"mlops"],
    "Kafka Connect/CDC": [r"debezium", r"\bcdc\b", r"kafka\s*connect"],
})
add("data", {
    "Spark": [r"(apache\s*)?spark", r"스파크", r"pyspark", r"spark\s*sql"],
    "Kafka": [r"(apache\s*)?kafka", r"카프카", r"msk"],
    "Airflow": [r"(apache\s*)?airflow", r"에어플로우", r"mwaa"],
    "Hadoop": [r"hadoop", r"하둡", r"hdfs", r"yarn", r"mapreduce"],
    "Hive": [r"hive", r"하이브"],
    "Flink": [r"flink", r"플링크"],
    "Trino/Presto": [r"trino", r"presto", r"athena"],
    "Iceberg": [r"iceberg"],
    "Delta Lake": [r"delta\s*lake"],
    "Databricks": [r"databricks"],
    "Snowflake": [r"snowflake"],
    "BigQuery": [r"bigquery", r"빅쿼리"],
    "Redshift": [r"redshift"],
    "AWS Glue": [r"glue"],
    "EMR": [r"emr"],
    "Kinesis": [r"kinesis"],
    "NiFi": [r"nifi"],
    "Logstash/Beats": [r"filebeat", r"beats"],
    "dbt": [r"dbt"],
    "ETL/ELT": [r"etl", r"elt", r"데이터\s*파이프라인", r"data\s*pipeline", r"(수집|적재|배치|스트리밍)\s*파이프라인"],
    "Data Warehouse": [r"data\s*warehouse", r"데이터\s*웨어하우스", r"\bdw\b", r"\bdwh\b"],
    "Data Lake": [r"data\s*lake", r"데이터\s*레이크", r"lakehouse", r"레이크\s*하우스"],
    "Data Mart": [r"data\s*mart", r"데이터\s*마트"],
    "데이터 거버넌스": [r"거버넌스", r"governance", r"메타\s*데이터\s*관리", r"데이터\s*카탈로그", r"data\s*catalog"],
    "데이터 품질": [r"데이터\s*품질", r"data\s*quality"],
    "스트리밍 처리": [r"실시간\s*(데이터\s*)?처리", r"스트리밍\s*처리", r"stream\s*processing"],
    "배치 처리": [r"배치\s*처리", r"batch\s*processing"],
    "Pandas": [r"pandas", r"판다스"],
    "NumPy": [r"numpy"],
    "Tableau": [r"tableau", r"태블로"],
    "Power BI": [r"power\s*bi"],
    "Superset/Redash": [r"superset", r"redash", r"metabase", r"looker"],
    "BI": [r"\bbi\b", r"비즈니스\s*인텔리전스"],
    "OLAP": [r"olap", r"druid", r"kudu"],
    "Jupyter": [r"jupyter", r"주피터"],
})
add("ai", {
    "머신러닝": [r"머신\s*러닝", r"machine\s*learning", r"\bml\b"],
    "딥러닝": [r"딥\s*러닝", r"deep\s*learning"],
    "PyTorch": [r"pytorch", r"파이토치"],
    "TensorFlow": [r"tensorflow", r"텐서플로", r"keras"],
    "scikit-learn": [r"scikit[\s-]*learn", r"sklearn"],
    "LLM": [r"llm", r"대규모\s*언어\s*모델", r"거대\s*언어\s*모델", r"gpt", r"claude", r"gemini", r"llama"],
    "RAG": [r"rag"],
    "LangChain": [r"langchain", r"랭체인", r"langgraph", r"llamaindex"],
    "AI Agent": [r"ai\s*agent", r"에이전트", r"agentic"],
    "생성형 AI": [r"생성형\s*ai", r"generative\s*ai", r"gen\s*ai"],
    "AI 코딩 도구": [r"copilot", r"cursor", r"claude\s*code", r"코파일럿", r"ai\s*(코딩|개발)\s*도구", r"ai\s*도구"],
    "AI": [r"ai(?!\s*(agent|도구|코딩|개발\s*도구))", r"인공\s*지능"],
    "MLflow/Kubeflow": [r"mlflow", r"kubeflow"],
    "NLP": [r"nlp", r"자연어\s*처리"],
    "Computer Vision": [r"computer\s*vision", r"컴퓨터\s*비전", r"opencv"],
})
add("collab", {
    "Git": [r"git(?!\s*(hub|lab|ops|flow))", r"깃(?!허브|랩)"],
    "GitHub": [r"github(?!\s*actions?)", r"깃허브"],
    "Bitbucket": [r"bitbucket"],
    "GitLab": [r"gitlab(?!\s*ci)", r"깃랩"],
    "Git Flow": [r"git\s*flow", r"깃\s*플로우"],
    "SVN": [r"svn"],
    "Jira": [r"jira", r"지라"],
    "Confluence": [r"confluence", r"컨플루언스"],
    "Notion": [r"notion", r"노션"],
    "Slack": [r"slack", r"슬랙"],
    "Agile/Scrum": [r"agile", r"애자일", r"scrum", r"스크럼", r"칸반", r"kanban"],
    "코드 리뷰": [r"코드\s*리뷰", r"code\s*review"],
    "Swagger/OpenAPI": [r"swagger", r"openapi"],
    "Postman": [r"postman"],
    "Kafka/MQ": [r"rabbitmq", r"activemq", r"sqs", r"메시지\s*큐", r"message\s*queue", r"\bmq\b", r"pub/sub"],
    "단위 테스트": [r"단위\s*테스트", r"unit\s*test", r"테스트\s*코드", r"통합\s*테스트"],
})
add("cs_cert", {
    "자료구조·알고리즘": [r"자료\s*구조", r"알고리즘", r"algorithm", r"data\s*structure"],
    "운영체제": [r"운영\s*체제", r"operating\s*system"],
    "컴퓨터 네트워크": [r"http(s)?(?!\w)", r"tcp", r"udp"],
    "객체지향": [r"객체\s*지향", r"oop", r"object[\s-]*oriented", r"solid"],
    "디자인 패턴": [r"디자인\s*패턴", r"design\s*pattern"],
    "정보처리기사": [r"정보\s*처리\s*기사", r"정처기"],
    "AWS 자격증": [r"aws\s*(saa|sap|sysops|solutions\s*architect|certified)", r"aws\s*자격"],
    "CKA/CKAD": [r"\bcka\b", r"\bckad\b", r"\bcks\b"],
    "네트워크 자격증": [r"ccna", r"ccnp", r"네트워크\s*관리사"],
    "리눅스 자격증": [r"리눅스\s*마스터", r"rhcsa", r"rhce", r"lpic"],
    "SQLD/SQLP": [r"sqld", r"sqlp", r"adsp", r"빅데이터\s*분석기사"],
    "보안 자격증": [r"정보\s*보안\s*기사", r"cissp", r"cisa"],
})

TECH = T
_EN = re.compile(r"[a-z0-9]")


def _wrap(p):
    return rf"(?<![a-z0-9+#.]){p}(?![a-z0-9+#])" if _EN.match(p) or p.startswith(r"\.") else p


COMPILED = [(canon, cat, re.compile("|".join(_wrap(p) for p in pats), re.I)) for canon, (cat, pats) in TECH.items()]
TECH_CATEGORY = {canon: cat for canon, (cat, _) in TECH.items()}
TOKEN = {canon: "T_" + re.sub(r"[^0-9A-Za-z가-힣+#]", "_", canon) for canon in TECH}   # 하나의 토큰으로 보존


def extract(text):
    if not text:
        return []
    t = text.lower()
    return sorted({c for c, _, rx in COMPILED if rx.search(t)})


def replace_with_tokens(text):
    """본문의 기술 표기를 표준 토큰으로 치환(형태소 분석 전). 긴 표기부터 치환해 복합 용어를 보존."""
    t = text
    for c, _, rx in sorted(COMPILED, key=lambda x: -max(len(p) for p in TECH[x[0]][1])):
        t = rx.sub(f" {TOKEN[c]} ", t)
    return t


def find_spans(text, canon):
    """KWIC 검토용: 해당 표준어의 표기가 나온 위치들."""
    rx = dict((c, r) for c, _, r in COMPILED)[canon]
    return [(m.start(), m.end()) for m in rx.finditer(text)]


if __name__ == "__main__":
    from collections import Counter
    print("표준어 수:", len(TECH))
    print(Counter(c for c, _ in TECH.values()))
