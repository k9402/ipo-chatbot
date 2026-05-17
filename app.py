import streamlit as st
import os
import datetime
import zipfile
import chardet
import numpy as np
import dataclasses
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import dart_fss as dart

import re
import pandas as pd     # 추가!
import requests

# ─── API 설정 ────────────────────────────────────────────
DART_API_KEY = st.secrets["DART_API_KEY"]
UPSTAGE_API_KEY = st.secrets["UPSTAGE_API_KEY"]

dart.set_api_key(api_key=DART_API_KEY)
client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1/solar")

# ─── 데이터 구조 ─────────────────────────────────────────
@dataclasses.dataclass
class IpoDeps:
    corp_name: str
    corp_code: str
    api_key: str
    rcept_no: str = None
    extracted_file_path: str = None
    raw_text: str = None
    chunks: list = dataclasses.field(default_factory=list)
    embedding_model: object = None
    chunk_embeddings: object = None

# ─── 캐시 함수 (무거운 작업은 한 번만) ───────────────────
@st.cache_resource
def load_corp_list():
    dart.set_api_key(api_key=DART_API_KEY)
    return dart.get_corp_list()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("jhgan/ko-sbert-multitask")

# ─── 핵심 로직 함수들 ────────────────────────────────────
@st.cache_data(ttl=3600)  # 1시간 동안 데이터 캐싱 (사이트 밴 방지)

def get_ipo_info_from_web(corp_name):
    """38.co.kr에서 특정 기업의 핵심 정보만 딕셔너리로 뽑아옵니다."""
    url = "http://www.38.co.kr/html/fund/index.htm?o=r"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'euc-kr'
        tables = pd.read_html(response.text)
        
        target_df = None
        for df in tables:
            if '종목명' in df.columns or (not df.empty and '종목명' in str(df.iloc[0].values)):
                if not '종목명' in df.columns:
                    df.columns = df.iloc[0]
                    df = df[1:]
                target_df = df
                break
                
        if target_df is not None:
            # 기업명으로 해당 줄 찾기
            result = target_df[target_df['종목명'].str.contains(corp_name[:2], na=False, case=False)]
            
            if not result.empty:
                # 첫 번째 매칭된 결과의 값을 딕셔너리로 저장해서 반환!
                row = result.iloc[0]
                return {
                    "종목명": str(row.get('종목명', '확인 불가')),
                    "수요예측일": str(row.get('수요예측일', '확인 불가')),
                    "희망공모가": str(row.get('희망공모가(원)', '확인 불가')),
                    "확정공모가": str(row.get('확정공모가', '확인 불가')),
                    "공모금액": str(row.get('공모금액(백만)', '확인 불가')) + "백만원",
                    "주간사": str(row.get('주간사', '확인 불가'))
                }
    except Exception as e:
        print(f"크롤링 에러: {e}")
        
    return None # 못 찾으면 None 반환



def split_text(text, chunk_size=1500, overlap=150):
    chunks = []
    start = 0
    current_section = "문서 시작 부분" # 기본 제목
    
    # '1.', 'I.', '제1부' 등 공시 문서의 목차 패턴을 찾는 공식
    section_pattern = re.compile(r'^(제\s*[1-9]+\s*부|I{1,3}V?\.|[1-9]+\.)\s+')

    while start < len(text):
        chunk_text = text[start:start + chunk_size]
        
        # 쪼갠 텍스트 안에서 목차 제목이 등장하면 업데이트
        for line in chunk_text.split('\n'):
            line = line.strip()
            # 40자 이내의 짧은 문장 중 목차 패턴과 일치하면 제목으로 저장!
            if len(line) < 40 and section_pattern.match(line):
                current_section = line
        
        # 텍스트와 목차 제목을 한 세트(딕셔너리)로 묶어서 보관
        chunks.append({
            "text": chunk_text,
            "section": current_section
        })
        start += (chunk_size - overlap)
    return chunks

def find_document(deps, crp_list):
    """DART에서 투자설명서 접수번호 찾기"""
    today = datetime.date.today()
    start_de = (today - datetime.timedelta(days=1095)).strftime('%Y%m%d')
    end_de = today.strftime('%Y%m%d')

    target_corp_obj = crp_list.find_by_corp_code(deps.corp_code)
    reports = target_corp_obj.search_filings(
        bgn_de=start_de, end_de=end_de, pblntf_ty='C', sort='date'
    )

    if not reports or not reports.report_list:
        return False

    reports.report_list.sort(key=lambda x: x.rcept_dt, reverse=True)

    targets = ['[정정]증권신고서', '증권신고서(지분증권)', '[발행조건확정]증권신고서']
    for report in reports.report_list:
        if '증권발행실적보고서' in report.report_nm:
            continue
        if any(t in report.report_nm for t in targets):
            deps.rcept_no = report.rcept_no
            return True
    return False

def download_and_parse(deps):
    """문서 다운로드 → XML 파싱 → 텍스트 추출"""
    download_path = './dart_files'
    os.makedirs(download_path, exist_ok=True)

    zip_path = dart.api.filings.download_document(
        path=download_path, rcept_no=deps.rcept_no
    )

    with zipfile.ZipFile(zip_path) as z:
        file_list = z.namelist()
        if not file_list:
            return False
        z.extractall(path=download_path)
        deps.extracted_file_path = os.path.join(download_path, file_list[0])

    rawdata = open(deps.extracted_file_path, "rb").read(10000)
    enc = chardet.detect(rawdata)['encoding'] or 'euc-kr'

    with open(deps.extracted_file_path, 'r', encoding=enc, errors='replace') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'lxml')
    for tag in soup(["script", "style"]):
        tag.decompose()
    deps.raw_text = soup.get_text(separator="\n", strip=True)
    return True

def embed_document(deps):
    """텍스트 청킹 + 임베딩"""
    model = load_embedding_model()
    deps.embedding_model = model
    deps.chunks = split_text(deps.raw_text)
    texts_to_embed = [chunk["text"] for chunk in deps.chunks]
    deps.chunk_embeddings = model.encode(texts_to_embed, normalize_embeddings=True)

def get_answer(question, deps, chat_history):
    """RAG로 관련 문서 검색 후 LLM 답변 생성"""
    query_vec = deps.embedding_model.encode([question], normalize_embeddings=True)
    similarities = np.dot(deps.chunk_embeddings, query_vec.T).flatten()
    top_indices = np.argsort(similarities)[-3:][::-1]

    context = "\n\n".join([deps.chunks[i]["text"] for i in top_indices])
    system_msg = {
        "role": "system",
        "content": (
            "너는 공모주 투자설명서를 분석해주는 친절한 전문가야. "
            "제공된 [문서 내용]에만 근거해서 답변해줘. "
            "문서에 없는 내용은 '문서에서 확인할 수 없습니다'라고 말해줘.\n\n"
            "## 답변 규칙\n"
            "1. 금융 초보자도 이해할 수 있도록 전문 용어를 쉬운 말로 풀어서 설명해줘.\n"
            "2. 답변 마지막에는 반드시 답변 내용과 관련된 어려운 용어나 투자 개념을 하나 골라서 "
            "꼬리질문을 유도해줘. 예시: '`청약경쟁률`이 뭔지 알려드릴까요?', "
            "'`주주배정 방식`이 궁금하신가요?' 형식으로 자연스럽게 끝내줘.\n\n"
            f"[문서 내용]\n{context}"
        )
    }

    messages = [system_msg] + chat_history + [{"role": "user", "content": question}]
    response = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=messages
    )
    answer = response.choices[0].message.content

    sources = []
    for i, idx in enumerate(top_indices):
        chunk_info = deps.chunks[idx]
        
        # 목차 이름 꺼내오기
        section_name = chunk_info["section"]
        # 텍스트 미리보기 (40자로 조금 짧게)
        hint = chunk_info["text"].strip()[:40].replace('\n', ' ')
        
        # 1위: [1. 모집의 개요] 공모가는 어쩌구... 형태로 완성!
        sources.append(f"{i+1}위: [{section_name}] {hint}...")

    return answer, sources

# ─── Streamlit UI ────────────────────────────────────────
st.set_page_config(
    page_title="AI 공모주 가이드",
    page_icon="📈",
    layout="wide"
)

# session_state 초기화
for key, default in [
    ("deps", None),
    ("chat_history", []),
    ("messages", []),
    ("initial_shown", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.title("📈 AI 공모주 가이드")
    st.caption("복잡한 공모주 자료, 챗봇으로 쉽게!")
    st.divider()

    st.subheader("🏢 기업 선택")

    with st.spinner("기업 목록 불러오는 중... (최초 1회만)"):
        crp_list = load_corp_list()

    corp_names = sorted([corp.corp_name for corp in crp_list.corps])

    search_input = st.text_input("기업명 검색", placeholder="예: 더본코리아, 케이뱅크")

    if search_input:
        filtered = [n for n in corp_names if search_input in n]
        if filtered:
            selected_corp = st.selectbox("검색 결과", filtered)

            if st.button("📂 투자설명서 불러오기", use_container_width=True, type="primary"):
                exact = crp_list.find_by_corp_name(selected_corp)
                exact_match = [c for c in exact if c.corp_name == selected_corp]

                if not exact_match:
                    st.error("❌ 기업을 찾을 수 없습니다.")
                else:
                    deps = IpoDeps(
                        corp_name=exact_match[0].corp_name,
                        corp_code=exact_match[0].corp_code,
                        api_key=DART_API_KEY
                    )

                    progress = st.progress(0, text="DART 문서 검색 중...")
                    try:
                        if not find_document(deps, crp_list):
                            st.error("❌ 투자설명서를 찾을 수 없습니다.")
                        else:
                            progress.progress(33, text="문서 다운로드 중...")
                            download_and_parse(deps)

                            progress.progress(66, text="AI 분석 준비 중...")
                            embed_document(deps)

                            progress.progress(100, text="완료!")
                            st.session_state.deps = deps
                            st.session_state.chat_history = []
                            st.session_state.messages = []
                            st.session_state.initial_shown = False
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ 오류 발생: {e}")
        else:
            st.warning("검색 결과가 없습니다.")

    if st.session_state.deps:
        st.divider()
        st.subheader("📊 분석 중인 기업")
        st.success(f"**{st.session_state.deps.corp_name}**")
        st.caption(f"문서 크기: {len(st.session_state.deps.raw_text):,}자")
        st.caption(f"청크 수: {len(st.session_state.deps.chunks)}개")

        if st.button("🔄 다른 기업 분석하기", use_container_width=True):
            st.session_state.deps = None
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

# ── 메인 화면 ─────────────────────────────────────────────
if not st.session_state.deps:
    # 시작 화면
    st.title("📈 AI 공모주 가이드")
    st.markdown("### 복잡한 공모주 자료, 챗봇으로 쉽게 알려드립니다!")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
**이런 분께 추천해요**
- 공모주에 관심 있지만 투자설명서가 너무 어려운 분
- 핵심 정보만 빠르게 파악하고 싶은 분
- 금융 용어가 낯선 초보 투자자
        """)
    with col2:
        st.success("""
**이런 질문을 할 수 있어요**
- 이 회사 대표가 누구야?
- 공모가가 얼마야?
- 위험요인이 뭐가 있어?
- 최대주주 지분이 몇 퍼센트야?
- 이 회사 매출은 얼마야?
        """)

    st.markdown("---")
    st.markdown("👈 **왼쪽에서 기업명을 검색해서 시작하세요!**")

else:
    # 챗봇 화면
    deps = st.session_state.deps
    st.title(f"📈 {deps.corp_name} 투자설명서 분석")

    # 기존 대화 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 참고한 문서 위치"):
                    for s in msg["sources"]:
                        st.caption(s)

    # 처음 접속 시 공모주 개요 자동 생성
    # if not st.session_state.initial_shown:
    #     dart_link = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={deps.rcept_no}"
    #     overview_prompt = (
    #         "이 공모주의 개요를 아래 항목으로 요약해줘: "
    #         "증권의 종류, 공모주식 수, 1주당 공모가격, 총 모집금액, 청약일정, 자금 사용목적, 주요 위험요인. "
    #         "각 항목을 번호 목록으로 정리하되, "
    #         "★주의: 제공된 문서 내용에 정확한 수치나 날짜가 없다면 절대로 지어내지 말고 '문서에서 검색되지 않음'이라고만 적어. "
    #         "금융 초보자도 이해할 수 있도록 쉬운 말로 설명해줘."
    #     )

    #     with st.chat_message("assistant"):
    #         with st.spinner("공모주 개요 분석 중..."):
    #             overview, sources = get_answer(overview_prompt, deps, [])

    #         greeting = (
    #             f"안녕하세요! 이 챗봇은 **{deps.corp_name}** 투자설명서를 분석해서 "
    #             f"공모가, 재무상태, 위험요인 등을 쉽게 알려드릴 수 있어요. "
    #             f"공모주에 대한 원본 공시 자료는 [DART에서 확인]({dart_link})해보세요.\n\n"
    #             f"---\n\n"
    #             f"**📋 {deps.corp_name} 공모주 개요**\n\n"
    #             f"{overview}\n\n"
    #             f"---\n\n"
    #             f"⚠️ *본 정보는 AI가 공시 자료를 바탕으로 요약한 것이며, "
    #             f"실제 투자 결정의 책임은 본인에게 있습니다.*"
    #         )

    #         st.markdown(greeting)
    #         with st.expander("📚 참고한 문서 위치"):
    #             for s in sources:
    #                 st.caption(s)

    #     st.session_state.messages.append({
    #         "role": "assistant",
    #         "content": greeting,
    #         "sources": sources
    #     })
    #     st.session_state.chat_history.append({"role": "assistant", "content": greeting})
    #     st.session_state.initial_shown = True

    if not st.session_state.initial_shown:
        dart_link = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={deps.rcept_no}"
        overview_prompt = (
            "이 공모주의 1. 청약일정, 2. 주요 위험요인을 요약해줘. "
            "각 항목을 번호 목록으로 정리하고, 금융 초보자도 이해할 수 있도록 쉬운 말로 설명해줘. "
            "문서에 정보가 없으면 지어내지 말고 '문서에서 확인할 수 없습니다'라고 적어."
        )
        with st.chat_message("assistant"):
            with st.spinner("공모주 핵심 정보 수집 및 분석 중..."):
                # 1. 크롤링으로 1~5번 데이터 가져오기
                web_info = get_ipo_info_from_web(deps.corp_name)
                
                # 2. AI로 6~7번(청약일정, 위험요인) 요약 가져오기
                overview_ai, sources = get_answer(overview_prompt, deps, [])

            # 3. 크롤링 텍스트 예쁘게 만들기
            if web_info:
                web_text = (
                    f"1. **종목명**: {web_info['종목명']}\n"
                    f"2. **수요예측일**: {web_info['수요예측일']}\n"
                    f"3. **희망공모가**: {web_info['희망공모가']}\n"
                    f"4. **확정공모가**: {web_info['확정공모가']}원\n"
                    f"5. **공모금액**: {web_info['공모금액']}\n"
                    f"6. **주간사**: {web_info['주간사']}\n"
                )
            else:
                web_text = "⚠️ 웹 크롤링을 통한 실시간 공모 정보(1~6번)를 불러오지 못했습니다.\n"

            # 4. 최종 인사말 완성 (크롤링 텍스트 + AI 텍스트)
            greeting = (
                f"안녕하세요! 이 챗봇은 **{deps.corp_name}** 투자설명서를 분석해서 "
                f"공모가, 재무상태, 위험요인 등을 쉽게 알려드릴 수 있어요. "
                f"공모주에 대한 원본 공시 자료는 [DART에서 확인]({dart_link})해보세요.\n\n"
                f"---\n\n"
                f"**📋 {deps.corp_name} 공모주 개요**\n\n"
                f"{web_text}\n"  # 크롤링 정보
                f"**AI 추가 분석 (DART 기준):**\n"
                f"{overview_ai}\n\n" # AI 정보
                f"---\n\n"
                f"⚠️ *본 정보는 웹 크롤링 및 AI 공시 자료 분석을 바탕으로 요약한 것이며, 실제 투자 결정의 책임은 본인에게 있습니다.*"
            )

            st.markdown(greeting)
            with st.expander("📚 참고한 문서 위치"):
                for s in sources:
                    st.caption(s)

        # 세션에 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting,
            "sources": sources
        })
        st.session_state.chat_history.append({"role": "assistant", "content": greeting})
        st.session_state.initial_shown = True




    # 채팅 입력
    if prompt := st.chat_input(f"{deps.corp_name}에 대해 궁금한 것을 물어보세요!"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("문서 분석 중..."):
                try:
                    answer, sources = get_answer(
                        prompt, deps, st.session_state.chat_history
                    )
                    st.write(answer)
                    with st.expander("📚 참고한 문서 위치"):
                        for s in sources:
                            st.caption(s)

                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"❌ 답변 생성 중 오류: {e}")
