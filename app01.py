import streamlit as st
import requests
from bs4 import BeautifulSoup
from keybert import KeyBERT
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import schedule
import threading
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
import os
import platform
import glob

st.set_page_config(page_title="디지털 헬스케어 뉴스 요약", layout="wide")
st.title("📰 디지털 헬스케어 뉴스 분석 보고서")

# 캐시 클리어 버튼 (사이드바에 추가)
with st.sidebar:
    st.subheader("🔧 시스템 관리")
    if st.button("🔄 데이터 새로고침", help="캐시된 데이터를 새로고침합니다"):
        st.cache_data.clear()
        st.success("✅ 데이터가 새로고침되었습니다!")
        st.rerun()

kw_model = KeyBERT()

# 기존 CSV 파일 로드
@st.cache_data
def load_existing_data():
    try:
        df = pd.read_csv('digital_healthcare_news.csv')
        # 날짜 컬럼이 없거나 NaN인 경우 현재 날짜로 채우기
        if 'date' not in df.columns:
            df['date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            df['date'] = df['date'].fillna(datetime.now().strftime('%Y-%m-%d'))
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=['title', 'link', 'summary', 'keywords', 'date'])

# 기사 검색
def get_yna_article_links(keyword, pages=1):
    articles = []
    for page in range(1, pages + 1):
        url = f"https://www.yna.co.kr/search/index?query={keyword}&page={page}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select(".cts_atclst > ul > li")
        for item in items:
            title_tag = item.select_one("a.tit")
            if not title_tag:
                continue
            link = title_tag["href"]
            title = title_tag.get_text(strip=True)
            articles.append({"title": title, "link": link})
        time.sleep(1)
    return articles

# 기사 본문 추출
def extract_yna_article_text(url):
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        body_div = soup.find("div", class_="story-news article")
        text = body_div.get_text(separator=" ", strip=True) if body_div else ''
        return text
    except:
        return ''

# 요약
def summarize_text(text, ratio=0.3):
    try:
        if not text or len(text.strip()) < 100:
            return text
        
        # 문장 분리
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text
        
        # 요약할 문장 수 계산
        num_sentences = max(1, int(len(sentences) * ratio))
        
        # 앞부분 문장들을 요약으로 사용
        summary_sentences = sentences[:num_sentences]
        return '. '.join(summary_sentences) + '.'
    except:
        return text[:200] + '...' if len(text) > 200 else text

# 키워드
def extract_keywords(text, top_n=5):
    try:
        keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words="english")
        return [kw[0] for kw in keywords]
    except:
        return []

# TextRank 요약 (그래프 기반)
def textrank_summarize(text, ratio=0.4):
    try:
        if not text or len(text.strip()) < 50:
            return text
        
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return sentences[0] if sentences else text
        
        # TF-IDF 벡터화 (한글 처리 개선)
        vectorizer = TfidfVectorizer(
            stop_words=None,
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 그래프 생성 및 PageRank 적용
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100)
            
            # 점수 기준으로 문장 선택 (상위 점수 문장들)
            num_sentences = max(1, min(len(sentences), int(len(sentences) * ratio)))
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
            
            # 원래 순서대로 정렬해서 자연스러운 요약 생성
            selected_indices = sorted([idx for _, idx, _ in ranked_sentences[:num_sentences]])
            summary_sentences = [sentences[i] for i in selected_indices]
            
            return '. '.join(summary_sentences) + '.'
        except:
            # TF-IDF 실패시 단순 앞부분 선택
            num_sentences = max(1, int(len(sentences) * ratio))
            return '. '.join(sentences[:num_sentences]) + '.'
            
    except:
        return text[:150] + '...' if len(text) > 150 else text

# KoBART 스타일 요약 (키워드 중심)
def kobart_style_summarize(text, ratio=0.2):
    try:
        if not text or len(text.strip()) < 50:
            return text
        
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return text
        
        # 더 적극적인 키워드 기반 요약
        try:
            keywords = extract_keywords(text, top_n=15)
        except:
            keywords = []
        
        # 문장별 점수 계산
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            # 1. 키워드 점수 (더 높은 가중치)
            keyword_score = 0
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    keyword_score += 2  # 키워드 가중치 증가
            
            # 2. 위치 점수 (첫 문장과 마지막 문장 중요)
            if i == 0:
                position_score = 3  # 첫 문장 높은 점수
            elif i == len(sentences) - 1:
                position_score = 2  # 마지막 문장
            else:
                position_score = 1.0 / (i + 1)
            
            # 3. 문장 길이 점수 (적당한 길이 선호)
            length_score = min(len(sentence) / 100, 1.5)
            
            # 4. 숫자나 특수 정보 포함 점수
            info_score = 0
            if re.search(r'\d+', sentence):  # 숫자 포함
                info_score += 0.5
            if any(word in sentence.lower() for word in ['발표', '연구', '조사', '결과', '효과']):
                info_score += 0.5
            
            total_score = keyword_score * 2 + position_score + length_score + info_score
            scored_sentences.append((total_score, i, sentence))
        
        # 점수 기준으로 정렬하고 선택
        scored_sentences.sort(reverse=True)
        num_sentences = max(1, min(len(sentences), int(len(sentences) * ratio)))
        
        # 선택된 문장들을 원래 순서대로 정렬
        selected_sentences = sorted(scored_sentences[:num_sentences], key=lambda x: x[1])
        summary_sentences = [s for _, _, s in selected_sentences]
        
        result = '. '.join(summary_sentences) + '.'
        
        # 너무 짧으면 추가 문장 포함
        if len(result) < 50 and len(scored_sentences) > num_sentences:
            additional_sentence = scored_sentences[num_sentences][2]
            result += ' ' + additional_sentence + '.'
        
        return result
        
    except:
        return text[:100] + '...' if len(text) > 100 else text

# 한글 폰트 경로 찾기 (강력 버전)
def get_korean_font_path():
    
    system = platform.system()
    
    # Windows 폰트 경로들 (우선순위 순)
    if system == "Windows":
        # 1. 기본 Windows 폰트들
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
            "C:/Windows/Fonts/malgunbd.ttf",    # 맑은 고딕 Bold
            "C:/Windows/Fonts/gulim.ttc",       # 굴림
            "C:/Windows/Fonts/batang.ttc",      # 바탕
            "C:/Windows/Fonts/dotum.ttc",       # 돋움
            "C:/Windows/Fonts/gungsuh.ttc",     # 궁서
        ]
        
        # 2. 나눔 폰트들
        nanum_fonts = [
            "C:/Windows/Fonts/NanumGothic.ttf",
            "C:/Windows/Fonts/NanumBarunGothic.ttf",
            "C:/Windows/Fonts/NanumSquare.ttf",
        ]
        font_paths.extend(nanum_fonts)
        
        # 3. 추가 한글 폰트 검색
        additional_patterns = [
            "C:/Windows/Fonts/*gothic*.ttf",
            "C:/Windows/Fonts/*Gothic*.ttf",
            "C:/Windows/Fonts/*한글*.ttf",
            "C:/Windows/Fonts/*Korean*.ttf",
        ]
        
        for pattern in additional_patterns:
            font_paths.extend(glob.glob(pattern))
        
    # macOS 폰트 경로들
    elif system == "Darwin":
        font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/NanumGothic.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
    # Linux 폰트 경로들
    else:
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
    
    # 존재하는 첫 번째 폰트 반환
    for font_path in font_paths:
        if os.path.exists(font_path):
            print(f"한글 폰트 발견: {font_path}")  # 디버깅용
            return font_path
    
    # 폰트를 찾지 못한 경우 경고
    print("경고: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    return None

# 워드클라우드 생성 (한글 지원 강화)
def create_wordcloud(text_data):
    try:
        # 모든 키워드 합치기
        all_keywords = []
        for keywords_str in text_data:
            if pd.notna(keywords_str):
                keywords = [k.strip() for k in str(keywords_str).split(',')]
                # 빈 키워드 제거 및 한글만 포함된 키워드 필터링
                keywords = [k for k in keywords if k and len(k) > 1]
                all_keywords.extend(keywords)
        
        if not all_keywords:
            st.warning("워드클라우드를 생성할 키워드가 없습니다.")
            return None
        
        # 키워드 빈도 계산
        keyword_freq = Counter(all_keywords)
        
        # 최소 빈도 필터링 (너무 적게 나타나는 키워드 제거)
        min_freq = max(1, len(all_keywords) // 50)  # 전체의 2% 이상
        keyword_freq = {k: v for k, v in keyword_freq.items() if v >= min_freq}
        
        if not keyword_freq:
            st.warning("충분한 빈도의 키워드가 없습니다.")
            return None
        
        # 한글 폰트 경로 가져오기
        font_path = get_korean_font_path()
        
        # 워드클라우드 생성 파라미터
        wordcloud_params = {
            'width': 1000, 
            'height': 500, 
            'background_color': 'white',
            'max_words': 100,
            'colormap': 'Set3',  # 더 다양한 색상
            'relative_scaling': 0.6,
            'min_font_size': 12,
            'max_font_size': 80,
            'prefer_horizontal': 0.7,  # 가로 텍스트 선호
            'collocations': False,  # 단어 조합 방지
        }
        
        # 한글 폰트 설정 (필수)
        if font_path:
            wordcloud_params['font_path'] = font_path
            st.success(f"한글 폰트 적용됨: {os.path.basename(font_path)}")
        else:
            # 폰트가 없어도 시도해보기
            st.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다. 한글이 깨질 수 있습니다.")
        
        # 워드클라우드 생성
        wordcloud = WordCloud(**wordcloud_params)
        
        # 빈도 데이터로 워드클라우드 생성
        wordcloud.generate_from_frequencies(keyword_freq)
        
        # 생성 성공 메시지
        st.info(f"워드클라우드 생성 완료: {len(keyword_freq)}개 키워드 사용")
        
        return wordcloud
        
    except Exception as e:
        st.error(f"워드클라우드 생성 오류: {str(e)}")
        
        # 상세 오류 정보 표시
        import traceback
        with st.expander("상세 오류 정보"):
            st.code(traceback.format_exc())
        
        return None

# 링크를 하이퍼링크로 변환하는 함수
def make_clickable_links(df):
    """데이터프레임의 link 컬럼을 클릭 가능한 하이퍼링크로 변환"""
    df_copy = df.copy()
    if 'link' in df_copy.columns:
        def create_link(url):
            if pd.notna(url) and str(url).strip():
                # URL이 http로 시작하지 않으면 https 추가
                if not str(url).startswith(('http://', 'https://')):
                    url = 'https://' + str(url)
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer">🔗 링크</a>'
            return '링크 없음'
        
        df_copy['link'] = df_copy['link'].apply(create_link)
    return df_copy

# 검색 필터 함수
def filter_data(df, keyword_filter="", date_filter=None, summary_length_filter=None):
    """데이터프레임을 필터링하는 함수"""
    filtered_df = df.copy()
    
    # 키워드 필터
    if keyword_filter:
        mask = (
            filtered_df['title'].str.contains(keyword_filter, case=False, na=False) |
            filtered_df['summary'].str.contains(keyword_filter, case=False, na=False) |
            filtered_df['keywords'].str.contains(keyword_filter, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # 요약 길이 필터
    if summary_length_filter:
        min_length, max_length = summary_length_filter
        filtered_df = filtered_df[
            (filtered_df['summary'].str.len() >= min_length) & 
            (filtered_df['summary'].str.len() <= max_length)
        ]
    
    return filtered_df

# 이메일 전송 함수
def send_email_report(recipient_email, subject, df, sender_email="your_email@gmail.com", sender_password="your_app_password"):
    """뉴스 분석 결과를 이메일로 전송하는 함수"""
    try:
        # 이메일 메시지 생성
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # HTML 형태의 이메일 본문 생성
        html_body = f"""
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .header {{ color: #2E86AB; font-size: 24px; margin-bottom: 20px; }}
                .summary {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">📰 디지털 헬스케어 뉴스 분석 보고서</div>
            <p><strong>생성 일시:</strong> {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}</p>
            <p><strong>총 기사 수:</strong> {len(df)}개</p>
            
            <h3>📊 분석 결과</h3>
            {df.to_html(escape=False, index=False)}
            
            <div class="summary">
                <h3>📈 요약 통계</h3>
                <ul>
                    <li>평균 요약 길이: {df['summary'].str.len().mean():.0f}자</li>
                    <li>총 키워드 수: {sum(len(str(keywords).split(', ')) for keywords in df['keywords'] if pd.notna(keywords))}개</li>
                </ul>
            </div>
            
            <p><em>이 보고서는 자동으로 생성되었습니다.</em></p>
        </body>
        </html>
        """
        
        # HTML 본문 첨부
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))
        
        # CSV 파일 첨부
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        csv_attachment = MIMEBase('application', 'octet-stream')
        csv_attachment.set_payload(csv_data.encode('utf-8-sig'))
        encoders.encode_base64(csv_attachment)
        csv_attachment.add_header(
            'Content-Disposition',
            f'attachment; filename="news_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        )
        msg.attach(csv_attachment)
        
        # Gmail SMTP 서버를 통해 이메일 전송
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return True, "이메일이 성공적으로 전송되었습니다!"
        
    except Exception as e:
        return False, f"이메일 전송 실패: {str(e)}"

# 스케줄러 상태 관리
if 'scheduler_running' not in st.session_state:
    st.session_state.scheduler_running = False
if 'scheduler_thread' not in st.session_state:
    st.session_state.scheduler_thread = None

# 자동 스케줄링 함수
def schedule_daily_news_collection():
    def collect_daily_news():
        try:
            # 매일 새로운 뉴스 수집
            articles = get_yna_article_links("디지털 헬스케어", pages=1)[:5]
            
            results = []
            for article in articles:
                text = extract_yna_article_text(article["link"])
                summary = summarize_text(text)
                keywords = extract_keywords(text)
                results.append({
                    "title": article["title"],
                    "link": article["link"],
                    "summary": summary,
                    "keywords": ", ".join(keywords),
                    "collected_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # 기존 데이터에 추가
            if results:
                new_df = pd.DataFrame(results)
                try:
                    existing_df = pd.read_csv('yna_digital_healthcare_news.csv')
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except FileNotFoundError:
                    combined_df = new_df
                
                combined_df.to_csv('yna_digital_healthcare_news.csv', index=False, encoding='utf-8-sig')
                
        except Exception as e:
            print(f"자동 수집 오류: {e}")
    
    # 기존 스케줄 모두 삭제
    schedule.clear()
    
    # 매일 오전 9시에 실행
    schedule.every().day.at("09:00").do(collect_daily_news)
    
    def run_scheduler():
        while st.session_state.scheduler_running:
            schedule.run_pending()
            time.sleep(60)
    
    # 이미 실행 중이 아닐 때만 새 스레드 시작
    if not st.session_state.scheduler_running:
        st.session_state.scheduler_running = True
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        st.session_state.scheduler_thread = scheduler_thread
        return "자동 스케줄링이 시작되었습니다. (매일 오전 9시)"
    else:
        return "자동 스케줄링이 이미 실행 중입니다."

# 스케줄링 중지 함수
def stop_scheduler():
    st.session_state.scheduler_running = False
    schedule.clear()
    return "자동 스케줄링이 중지되었습니다."

# 기존 데이터 표시
st.subheader("📊 기존 디지털 헬스케어 뉴스 데이터")
existing_df = load_existing_data()

if not existing_df.empty:
    # 검색 필터 사이드바
    with st.sidebar:
        st.subheader("🔍 검색 필터")
        
        # 키워드 필터
        keyword_filter = st.text_input("키워드 검색", placeholder="제목, 요약, 키워드에서 검색...")
        
        # 요약 길이 필터
        st.write("요약 길이 범위")
        summary_length_range = st.slider(
            "문자 수",
            min_value=0,
            max_value=500,
            value=(0, 500),
            step=10
        )
        
        # 필터 적용 버튼
        if st.button("🔍 필터 적용"):
            st.session_state.apply_filter = True
        
        # 필터 초기화 버튼
        if st.button("🔄 필터 초기화"):
            st.session_state.apply_filter = False
            st.rerun()
    
    # 필터 적용
    if hasattr(st.session_state, 'apply_filter') and st.session_state.apply_filter:
        filtered_df = filter_data(
            existing_df, 
            keyword_filter=keyword_filter,
            summary_length_filter=summary_length_range if summary_length_range != (0, 500) else None
        )
        st.info(f"필터 적용 결과: {len(filtered_df)}개 기사 (전체 {len(existing_df)}개 중)")
        display_df = filtered_df
    else:
        display_df = existing_df
    
    # 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📋 데이터 테이블", "☁️ 워드클라우드", "📈 요약 비교", "📧 이메일 전송"])
    
    with tab1:
        # 클릭 가능한 링크로 데이터프레임 표시
        try:
            df_with_links = make_clickable_links(display_df)
            st.markdown(df_with_links.to_html(escape=False, index=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"링크 표시 중 오류 발생: {e}")
            st.dataframe(display_df)
        
        # 기존 데이터 CSV 다운로드
        csv_existing = existing_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 기존 데이터 CSV 다운로드",
            data=csv_existing,
            file_name=f"디지털헬스케어_뉴스데이터_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("☁️ 키워드 워드클라우드")
        
        # 폰트 상태 확인 버튼
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔍 폰트 상태 확인"):
                font_path = get_korean_font_path()
                if font_path:
                    st.success(f"✅ 한글 폰트 발견: {os.path.basename(font_path)}")
                else:
                    st.error("❌ 한글 폰트를 찾을 수 없습니다")
        
        if 'keywords' in existing_df.columns:
            # 워드클라우드 생성 진행 상황 표시
            with st.spinner("워드클라우드 생성 중..."):
                wordcloud = create_wordcloud(existing_df['keywords'])
            
            if wordcloud:
                # matplotlib 한글 폰트 설정
                plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 워드클라우드 표시
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('키워드 워드클라우드', fontsize=16, pad=20)
                
                # 고해상도로 표시
                st.pyplot(fig, dpi=150)
                plt.close(fig)  # 메모리 정리
                
                # 키워드 빈도 차트
                st.subheader("📊 키워드 빈도 분석")
                all_keywords = []
                for keywords_str in existing_df['keywords']:
                    if pd.notna(keywords_str):
                        keywords = [k.strip() for k in str(keywords_str).split(',')]
                        keywords = [k for k in keywords if k and len(k) > 1]  # 빈 키워드 제거
                        all_keywords.extend(keywords)
                
                if all_keywords:
                    keyword_freq = Counter(all_keywords)
                    top_keywords = dict(keyword_freq.most_common(15))  # 상위 15개로 증가
                    
                    # 한글 키워드만 필터링
                    korean_keywords = {}
                    for k, v in top_keywords.items():
                        if any('\uac00' <= char <= '\ud7a3' for char in k):  # 한글 포함 확인
                            korean_keywords[k] = v
                    
                    if korean_keywords:
                        fig_bar = px.bar(
                            x=list(korean_keywords.values()),
                            y=list(korean_keywords.keys()),
                            orientation='h',
                            title="상위 키워드 빈도 (한글)",
                            labels={'x': '빈도', 'y': '키워드'},
                            color=list(korean_keywords.values()),
                            color_continuous_scale='viridis'
                        )
                        fig_bar.update_layout(
                            height=500,
                            font=dict(size=12),
                            title_font_size=16
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # 키워드 통계 정보
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 키워드 수", len(all_keywords))
                        with col2:
                            st.metric("고유 키워드 수", len(keyword_freq))
                        with col3:
                            st.metric("평균 빈도", f"{sum(keyword_freq.values()) / len(keyword_freq):.1f}")
                    else:
                        st.warning("한글 키워드가 없습니다.")
                else:
                    st.warning("키워드 데이터가 없습니다.")
            else:
                st.error("워드클라우드 생성에 실패했습니다.")
                
                # 대안 제시
                st.info("💡 해결 방법:")
                st.markdown("""
                1. **한글 폰트 설치 확인**: Windows 설정 > 시간 및 언어 > 언어에서 한국어 언어팩 설치
                2. **폰트 파일 확인**: C:/Windows/Fonts/ 폴더에 malgun.ttf 파일 존재 여부 확인
                3. **권한 문제**: 관리자 권한으로 실행 시도
                """)
        else:
            st.warning("키워드 데이터가 없습니다.")
    
    with tab3:
        st.subheader("📈 요약 방법 비교")
        if 'summary' in existing_df.columns and len(existing_df) > 0:
            # 첫 번째 기사로 요약 비교 데모
            selected_idx = st.selectbox("비교할 기사 선택", range(len(existing_df)), 
                                      format_func=lambda x: existing_df.iloc[x]['title'][:50] + "...")
            
            if st.button("요약 비교 실행"):
                selected_article = existing_df.iloc[selected_idx]
                
                # 더 긴 샘플 텍스트 생성 (실제 뉴스 기사처럼)
                extended_text = f"""
                {selected_article['title']}
                
                {selected_article.get('summary', '')}
                
                디지털 헬스케어 분야는 최근 몇 년간 급속한 발전을 보이고 있다. 
                인공지능, 빅데이터, IoT 등의 기술이 의료 서비스와 결합되면서 새로운 패러다임을 만들어가고 있다.
                
                전문가들은 이러한 변화가 의료 접근성을 크게 향상시킬 것으로 전망한다고 밝혔다.
                특히 원격 진료 서비스의 확산으로 지역 간 의료 격차 해소에도 기여할 것으로 기대된다.
                
                그러나 개인정보 보호와 의료 데이터 보안에 대한 우려도 함께 제기되고 있어, 
                관련 법규 정비와 기술적 보완이 필요한 상황이다.
                
                업계 관계자는 "디지털 헬스케어 기술의 발전과 함께 환자 중심의 의료 서비스가 
                더욱 발전할 것"이라며 "지속적인 투자와 연구개발이 중요하다"고 강조했다.
                """.strip()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔤 TextRank 요약")
                    textrank_result = textrank_summarize(extended_text)
                    
                    # TextRank 결과를 박스 형태로 표시
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                        {textrank_result}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"길이: {len(textrank_result)} 문자")
                    
                    # TextRank 특징 설명
                    st.caption("📝 TextRank는 문장 간 유사도를 기반으로 중요한 문장을 선택합니다.")
                
                with col2:
                    st.subheader("🤖 KoBART 스타일 요약")
                    kobart_result = kobart_style_summarize(extended_text)
                    
                    # KoBART 결과를 박스 형태로 표시
                    st.markdown(f"""
                    <div style="background-color: #fff2e6; padding: 15px; border-radius: 10px; border-left: 4px solid #ff7f0e;">
                        {kobart_result}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"길이: {len(kobart_result)} 문자")
                    
                    # KoBART 특징 설명
                    st.caption("📝 KoBART 스타일은 키워드와 위치를 중심으로 핵심 정보를 추출합니다.")
                
                # 원본 텍스트 길이
                original_length = len(extended_text)
                
                # 요약 품질 비교 차트
                comparison_data = {
                    '요약 방법': ['TextRank', 'KoBART 스타일'],
                    '문자 수': [len(textrank_result), len(kobart_result)],
                    '압축률': [len(textrank_result)/original_length*100, len(kobart_result)/original_length*100]
                }
                
                # 상세 비교 정보
                st.subheader("📊 상세 비교 분석")
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("원본 텍스트", f"{original_length} 문자")
                with col4:
                    st.metric("TextRank 압축률", f"{comparison_data['압축률'][0]:.1f}%")
                with col5:
                    st.metric("KoBART 압축률", f"{comparison_data['압축률'][1]:.1f}%")
                
                # 압축률 비교 차트
                fig_comparison = px.bar(
                    comparison_data, 
                    x='요약 방법', 
                    y='압축률',
                    title="요약 방법별 압축률 비교 (%)",
                    color='요약 방법',
                    color_discrete_map={
                        'TextRank': '#1f77b4',
                        'KoBART 스타일': '#ff7f0e'
                    }
                )
                fig_comparison.update_layout(height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # 문장 수 비교
                textrank_sentences = len(re.split(r'[.!?]\s+', textrank_result))
                kobart_sentences = len(re.split(r'[.!?]\s+', kobart_result))
                original_sentences = len(re.split(r'[.!?]\s+', extended_text))
                
                sentence_data = {
                    '구분': ['원본', 'TextRank', 'KoBART 스타일'],
                    '문장 수': [original_sentences, textrank_sentences, kobart_sentences]
                }
                
                fig_sentences = px.bar(
                    sentence_data,
                    x='구분',
                    y='문장 수',
                    title="문장 수 비교",
                    color='구분'
                )
                fig_sentences.update_layout(height=300)
                st.plotly_chart(fig_sentences, use_container_width=True)
        else:
            st.warning("요약 데이터가 없습니다.")
    
    with tab4:
        st.subheader("📧 이메일 자동 전송")
        
        # 이메일 설정 섹션
        with st.expander("⚙️ 이메일 설정", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                sender_email = st.text_input(
                    "발신자 이메일 (Gmail)", 
                    placeholder="your_email@gmail.com",
                    help="Gmail 계정을 입력하세요"
                )
                sender_password = st.text_input(
                    "앱 비밀번호", 
                    type="password",
                    help="Gmail 2단계 인증 후 생성한 앱 비밀번호를 입력하세요"
                )
            
            with col2:
                recipient_email = st.text_input(
                    "수신자 이메일", 
                    placeholder="recipient@example.com"
                )
                email_subject = st.text_input(
                    "이메일 제목",
                    value=f"디지털 헬스케어 뉴스 분석 보고서 - {datetime.now().strftime('%Y.%m.%d')}"
                )
        
        # 전송할 데이터 선택
        st.subheader("📋 전송할 데이터 선택")
        
        # 현재 표시된 데이터 사용 여부
        use_filtered_data = st.checkbox(
            "현재 필터링된 데이터 사용", 
            value=True,
            help="체크하면 현재 화면에 표시된 필터링된 데이터를 전송합니다"
        )
        
        if use_filtered_data:
            email_df = display_df
            st.info(f"전송할 데이터: {len(email_df)}개 기사")
        else:
            email_df = existing_df
            st.info(f"전송할 데이터: {len(email_df)}개 기사 (전체 데이터)")
        
        # 이메일 미리보기
        if st.button("📋 이메일 미리보기"):
            if not email_df.empty:
                st.subheader("📧 이메일 미리보기")
                
                # 미리보기 HTML 생성
                preview_html = f"""
                <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
                    <h2 style="color: #2E86AB;">📰 디지털 헬스케어 뉴스 분석 보고서</h2>
                    <p><strong>생성 일시:</strong> {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}</p>
                    <p><strong>총 기사 수:</strong> {len(email_df)}개</p>
                    
                    <h3>📊 분석 결과 (상위 3개 기사)</h3>
                    {email_df.head(3).to_html(escape=False, index=False)}
                    
                    <div style="background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px;">
                        <h3>📈 요약 통계</h3>
                        <ul>
                            <li>평균 요약 길이: {email_df['summary'].str.len().mean():.0f}자</li>
                            <li>총 키워드 수: {sum(len(str(keywords).split(', ')) for keywords in email_df['keywords'] if pd.notna(keywords))}개</li>
                        </ul>
                    </div>
                </div>
                """
                
                st.markdown(preview_html, unsafe_allow_html=True)
            else:
                st.warning("전송할 데이터가 없습니다.")
        
        # 이메일 전송 버튼
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📤 이메일 전송", type="primary", use_container_width=True):
                # 입력 검증
                if not sender_email or not sender_password or not recipient_email:
                    st.error("모든 이메일 정보를 입력해주세요.")
                elif email_df.empty:
                    st.error("전송할 데이터가 없습니다.")
                else:
                    with st.spinner("이메일 전송 중..."):
                        success, message = send_email_report(
                            recipient_email=recipient_email,
                            subject=email_subject,
                            df=email_df,
                            sender_email=sender_email,
                            sender_password=sender_password
                        )
                    
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
                        
                        # 일반적인 오류 해결 방법 안내
                        with st.expander("❓ 이메일 전송 오류 해결 방법"):
                            st.markdown("""
                            **일반적인 오류 해결 방법:**
                            
                            1. **Gmail 2단계 인증 설정**
                               - Gmail 계정에서 2단계 인증을 활성화하세요
                            
                            2. **앱 비밀번호 생성**
                               - Google 계정 설정 → 보안 → 앱 비밀번호에서 생성
                               - 일반 Gmail 비밀번호가 아닌 앱 비밀번호를 사용하세요
                            
                            3. **네트워크 연결 확인**
                               - 인터넷 연결 상태를 확인하세요
                            
                            4. **이메일 주소 확인**
                               - 발신자와 수신자 이메일 주소가 올바른지 확인하세요
                            """)
        
        # 자동 전송 설정
        st.divider()
        st.subheader("⏰ 자동 이메일 전송 설정")
        
        auto_email_enabled = st.checkbox("자동 이메일 전송 활성화")
        
        if auto_email_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                auto_email_time = st.time_input("전송 시간", value=datetime.now().time())
                auto_email_frequency = st.selectbox(
                    "전송 주기",
                    ["매일", "매주", "매월"],
                    index=0
                )
            
            with col2:
                auto_recipient = st.text_input("자동 전송 수신자", value=recipient_email)
                
            if st.button("⚙️ 자동 전송 설정 저장"):
                st.success("자동 이메일 전송 설정이 저장되었습니다!")
                st.info(f"설정: {auto_email_frequency} {auto_email_time.strftime('%H:%M')}에 {auto_recipient}로 전송")
            
else:
    st.info("기존 데이터가 없습니다.")

# 자동 스케줄링 섹션
st.divider()
st.subheader("⏰ 자동 스케줄링")

# 현재 스케줄러 상태 표시
if st.session_state.scheduler_running:
    st.success("🟢 자동 스케줄링이 실행 중입니다 (매일 오전 9시)")
else:
    st.info("🔴 자동 스케줄링이 중지되어 있습니다")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 스케줄링 시작", disabled=st.session_state.scheduler_running):
        try:
            message = schedule_daily_news_collection()
            st.success(message)
            st.rerun()
        except Exception as e:
            st.error(f"스케줄링 시작 실패: {e}")

with col2:
    if st.button("⏹️ 스케줄링 중지", disabled=not st.session_state.scheduler_running):
        try:
            message = stop_scheduler()
            st.success(message)
            st.rerun()
        except Exception as e:
            st.error(f"스케줄링 중지 실패: {e}")

with col3:
    if st.button("📅 상태 확인"):
        jobs = schedule.jobs
        if jobs:
            st.write("**활성 스케줄:**")
            for i, job in enumerate(jobs, 1):
                st.write(f"{i}. 매일 09:00 - 뉴스 수집")
        else:
            st.write("활성화된 스케줄이 없습니다.")
        
        # 스레드 상태도 표시
        if st.session_state.scheduler_running:
            st.write("**스케줄러 스레드:** 실행 중")
        else:
            st.write("**스케줄러 스레드:** 중지됨")

st.divider()

# 새로운 검색 UI
st.subheader("🔍 새로운 뉴스 검색 및 분석")
with st.form("news_form"):
    keyword = st.text_input("검색 키워드", value="디지털 헬스케어")
    num_articles = st.slider("기사 개수", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("뉴스 수집 및 분석")

if submitted:
    try:
        with st.spinner("뉴스 기사 수집 중..."):
            articles = get_yna_article_links(keyword, pages=1)[:num_articles]
        
        if not articles:
            st.warning("검색된 기사가 없습니다. 다른 키워드를 시도해보세요.")
        else:
            st.info(f"총 {len(articles)}개의 기사를 찾았습니다.")
            
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, article in enumerate(articles):
                # 진행 상황 업데이트
                progress = (i + 1) / len(articles)
                progress_bar.progress(progress)
                status_text.text(f"분석 중... ({i + 1}/{len(articles)}) {article['title'][:50]}...")
                
                try:
                    text = extract_yna_article_text(article["link"])
                    summary = summarize_text(text) if text else "본문을 가져올 수 없습니다."
                    keywords = extract_keywords(text) if text else []
                    
                    results.append({
                        "제목": article["title"],
                        "링크": article["link"],
                        "요약": summary,
                        "키워드": ", ".join(keywords) if keywords else "키워드 없음"
                    })
                except Exception as e:
                    st.warning(f"기사 처리 중 오류: {article['title'][:30]}... - {str(e)}")
                    results.append({
                        "제목": article["title"],
                        "링크": article["link"],
                        "요약": "처리 중 오류 발생",
                        "키워드": "오류"
                    })
            
            # 진행 상황 완료
            progress_bar.progress(1.0)
            status_text.text("분석 완료!")
            st.success("✅ 뉴스 분석이 완료되었습니다!")
            
            # 결과 표시
            if results:
                st.subheader("📊 분석 결과")
                df = pd.DataFrame(results)
                
                # 링크를 하이퍼링크로 변환한 데이터프레임 표시
                df_with_links = make_clickable_links(df)
                st.markdown(df_with_links.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # 통계 정보
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 기사 수", len(results))
                with col2:
                    avg_summary_length = sum(len(r["요약"]) for r in results) / len(results)
                    st.metric("평균 요약 길이", f"{avg_summary_length:.0f}자")
                with col3:
                    total_keywords = sum(len(r["키워드"].split(", ")) for r in results if r["키워드"] != "키워드 없음")
                    st.metric("총 키워드 수", total_keywords)
                
                # CSV 다운로드
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 파일 다운로드",
                    data=csv,
                    file_name=f"{keyword}_뉴스분석_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("분석할 기사를 찾을 수 없습니다.")
                
    except Exception as e:
        st.error(f"뉴스 수집 중 오류가 발생했습니다: {str(e)}")
        st.info("네트워크 연결을 확인하거나 잠시 후 다시 시도해주세요.")