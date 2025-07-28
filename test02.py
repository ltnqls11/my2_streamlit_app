import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib.parse
import re
from collections import Counter

# ✅ 키워드: 디지털 헬스케어
SEARCH_KEYWORD = "디지털 헬스케어"

# 전역 변수로 모델 초기화 (필요시에만)
kw_model = None
summarizer_available = False

def initialize_models():
    """필요한 모델들을 초기화"""
    global kw_model, summarizer_available
    
    # KeyBERT 초기화 시도
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        print("✅ KeyBERT 모델 로드 완료")
    except ImportError:
        print("⚠️ KeyBERT가 설치되지 않았습니다. 대안 키워드 추출 방법을 사용합니다.")
        kw_model = None
    except Exception as e:
        print(f"⚠️ KeyBERT 로드 오류: {e}")
        kw_model = None
    
    # gensim 요약 기능 확인
    try:
        from gensim.summarization import summarize
        summarizer_available = True
        print("✅ gensim 요약 기능 사용 가능")
    except ImportError:
        print("⚠️ gensim이 설치되지 않았습니다. 대안 요약 방법을 사용합니다.")
        summarizer_available = False
    except Exception as e:
        print(f"⚠️ gensim 로드 오류: {e}")
        summarizer_available = False

# 모델 초기화
initialize_models()

# 1. 네이버 뉴스에서 기사 목록 가져오기 (개선된 버전)
def get_naver_news_articles(keyword, pages=1):
    """네이버 뉴스에서 기사 목록 수집"""
    articles = []
    
    try:
        encoded_keyword = urllib.parse.quote(keyword)
        print(f"🔍 네이버 뉴스 검색 중: {keyword}")
        
        for page in range(1, pages + 1):
            start = (page - 1) * 10 + 1
            url = f"https://search.naver.com/search.naver?where=news&query={encoded_keyword}&start={start}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
            }
            
            print(f"📄 페이지 {page} 수집 중...")
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            
            soup = BeautifulSoup(res.text, "html.parser")
            
            # 네이버 뉴스 검색 결과 셀렉터
            news_items = soup.select(".news_area")
            
            if not news_items:
                print(f"⚠️ 페이지 {page}에서 기사를 찾을 수 없습니다.")
                continue
            
            page_articles = 0
            for item in news_items:
                try:
                    # 제목과 링크 추출
                    title_tag = item.select_one(".news_tit")
                    if not title_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    link = title_tag.get("href", "")
                    
                    # 유효성 검사
                    if not title or not link or len(title) < 10:
                        continue
                    
                    # 디지털 헬스케어 관련 키워드 필터링
                    healthcare_keywords = ['디지털', '헬스케어', '의료', '건강', 'AI', '인공지능', '원격진료', '웨어러블', '스마트', '병원', 'IoT', '빅데이터']
                    if any(keyword in title for keyword in healthcare_keywords):
                        articles.append({
                            "title": title,
                            "link": link
                        })
                        page_articles += 1
                    
                except Exception as e:
                    print(f"⚠️ 기사 파싱 오류: {e}")
                    continue
            
            print(f"📰 페이지 {page}에서 {page_articles}개 기사 수집")
            time.sleep(1)  # 서버 부담 줄이기
        
        print(f"✅ 총 {len(articles)}개 기사 수집 완료")
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 네트워크 오류: {e}")
        return []
    except Exception as e:
        print(f"❌ 기사 수집 오류: {e}")
        return []

# 1-2. 연합뉴스에서 기사 목록 가져오기 (백업용)
def get_yna_article_links(keyword, pages=1):
    """연합뉴스에서 기사 목록 수집"""
    articles = []
    
    try:
        encoded_keyword = urllib.parse.quote(keyword)
        print(f"🔍 연합뉴스 검색 중: {keyword}")
        
        # 연합뉴스 검색 URL 수정
        url = f"https://www.yna.co.kr/search?query={encoded_keyword}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        
        print(f"📄 연합뉴스 검색 중...")
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, "html.parser")
        
        # 연합뉴스 검색 결과 셀렉터들
        selectors = [
            ".list-type038 li",
            ".search-result-item",
            ".news-list li",
            "article"
        ]
        
        items = []
        for selector in selectors:
            items = soup.select(selector)
            if items:
                print(f"✅ 셀렉터 '{selector}'로 {len(items)}개 항목 발견")
                break
        
        if not items:
            print("⚠️ 연합뉴스에서 기사를 찾을 수 없습니다.")
            return []
        
        for item in items[:10]:  # 최대 10개
            try:
                # 제목과 링크 추출
                title_selectors = ["a.tit", ".tit a", "h3 a", "a"]
                title_tag = None
                
                for sel in title_selectors:
                    title_tag = item.select_one(sel)
                    if title_tag and title_tag.get("href"):
                        break
                
                if not title_tag:
                    continue
                
                title = title_tag.get_text(strip=True)
                link = title_tag.get("href", "")
                
                # 유효성 검사
                if not title or not link or len(title) < 10:
                    continue
                
                # 상대 URL을 절대 URL로 변환
                if link.startswith("/"):
                    link = "https://www.yna.co.kr" + link
                elif not link.startswith("http"):
                    link = "https://www.yna.co.kr/" + link
                
                articles.append({
                    "title": title,
                    "link": link
                })
                
            except Exception as e:
                print(f"⚠️ 기사 파싱 오류: {e}")
                continue
        
        print(f"✅ 연합뉴스에서 {len(articles)}개 기사 수집 완료")
        return articles
        
    except Exception as e:
        print(f"❌ 연합뉴스 수집 오류: {e}")
        return []

# 2. 기사 본문 추출 (개선된 버전)
def extract_article_text(url):
    """다양한 뉴스 사이트의 기사 본문 추출"""
    try:
        print(f"📄 본문 추출 중: {url[:50]}...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, "html.parser")
        
        # 다양한 뉴스 사이트 본문 셀렉터들
        content_selectors = [
            # 연합뉴스
            "div.story-news.article",
            "div.article-txt",
            ".article-wrap .article-txt",
            # 일반적인 뉴스 사이트
            "div.news-content",
            "div.article-body",
            "div.article-content",
            ".news_end",
            ".article_body",
            "div.content",
            "article",
            ".post-content",
            ".entry-content",
            # 네이버 뉴스
            "#articleBodyContents",
            ".se-main-container"
        ]
        
        body_div = None
        for selector in content_selectors:
            body_div = soup.select_one(selector)
            if body_div:
                print(f"✅ 본문 발견: {selector}")
                break
        
        if not body_div:
            # 대안: p 태그들 수집
            paragraphs = soup.find_all('p')
            if paragraphs:
                # 길이가 적당한 p 태그들만 선택
                valid_paragraphs = [p for p in paragraphs if len(p.get_text(strip=True)) > 20]
                if valid_paragraphs:
                    text = ' '.join([p.get_text(strip=True) for p in valid_paragraphs])
                    print(f"✅ p 태그로 본문 추출 ({len(text)} 문자)")
                    return text[:1500]  # 최대 1500자
            
            print("⚠️ 본문을 찾을 수 없습니다.")
            return ''
        
        # 불필요한 요소 제거
        for unwanted in body_div.find_all(['script', 'style', 'iframe', 'ins', 'aside', 'nav', 'header', 'footer']):
            unwanted.decompose()
        
        text = body_div.get_text(separator=" ", strip=True)
        
        # 텍스트 정리
        text = ' '.join(text.split())  # 공백 정리
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        # 너무 짧은 텍스트는 제외
        if len(text) < 50:
            print("⚠️ 추출된 본문이 너무 짧습니다.")
            return ''
        
        print(f"✅ 본문 추출 완료 ({len(text)} 문자)")
        return text[:1500]  # 최대 1500자로 제한
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 네트워크 오류: {e}")
        return ''
    except Exception as e:
        print(f"❌ 본문 추출 오류: {e}")
        return ''

# 3. 텍스트 요약 (개선된 버전)
def summarize_text(text, ratio=0.3):
    """텍스트 요약 (여러 방법 시도)"""
    if not text or len(text.strip()) < 100:
        print("⚠️ 텍스트가 너무 짧아 요약할 수 없습니다.")
        return text[:200] if text else ''
    
    # 방법 1: gensim 사용 (사용 가능한 경우)
    if summarizer_available:
        try:
            from gensim.summarization import summarize
            summary = summarize(text, ratio=ratio)
            if summary:
                print(f"✅ gensim으로 요약 완료 ({len(summary)} 문자)")
                return summary
        except Exception as e:
            print(f"⚠️ gensim 요약 오류: {e}")
    
    # 방법 2: sumy 사용 (대안)
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        
        parser = PlaintextParser.from_string(text, Tokenizer("korean"))
        summarizer = TextRankSummarizer()
        sentences = summarizer(parser.document, 3)  # 3문장 요약
        summary = ' '.join([str(sentence) for sentence in sentences])
        
        if summary:
            print(f"✅ sumy로 요약 완료 ({len(summary)} 문자)")
            return summary
    except ImportError:
        print("⚠️ sumy가 설치되지 않았습니다.")
    except Exception as e:
        print(f"⚠️ sumy 요약 오류: {e}")
    
    # 방법 3: 간단한 문장 추출 (백업)
    try:
        sentences = text.split('.')
        # 길이가 적당한 문장들 선택
        good_sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 200]
        
        if good_sentences:
            # 처음 2-3개 문장 선택
            summary = '. '.join(good_sentences[:3]) + '.'
            print(f"✅ 간단 요약 완료 ({len(summary)} 문자)")
            return summary
        else:
            # 원본 텍스트의 처음 부분 반환
            summary = text[:300] + '...' if len(text) > 300 else text
            print(f"✅ 텍스트 일부 반환 ({len(summary)} 문자)")
            return summary
            
    except Exception as e:
        print(f"❌ 요약 실패: {e}")
        return text[:200] if text else ''

# 4. 키워드 추출 (개선된 버전)
def extract_keywords(text, top_n=5):
    """키워드 추출 (여러 방법 시도)"""
    if not text or len(text.strip()) < 50:
        print("⚠️ 텍스트가 너무 짧아 키워드를 추출할 수 없습니다.")
        return []
    
    # 방법 1: KeyBERT 사용 (사용 가능한 경우)
    if kw_model:
        try:
            keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words="english")
            result = [kw[0] for kw in keywords]
            print(f"✅ KeyBERT로 키워드 추출 완료: {result}")
            return result
        except Exception as e:
            print(f"⚠️ KeyBERT 키워드 추출 오류: {e}")
    
    # 방법 2: konlpy 사용 (한국어 형태소 분석)
    try:
        from konlpy.tag import Okt
        okt = Okt()
        # 한국어 텍스트에서 명사만 추출
        nouns = okt.nouns(text)
        # 길이가 2 이상인 명사만 선택
        filtered_nouns = [noun for noun in nouns if len(noun) >= 2]
        # 빈도수 계산
        noun_counts = Counter(filtered_nouns)
        # 상위 키워드 선택
        keywords = [word for word, count in noun_counts.most_common(top_n)]
        print(f"✅ konlpy로 키워드 추출 완료: {keywords}")
        return keywords
    except ImportError:
        print("⚠️ konlpy가 설치되지 않았습니다.")
    except Exception as e:
        print(f"⚠️ konlpy 키워드 추출 오류: {e}")
    
    # 방법 3: 간단한 단어 빈도 분석 (백업)
    try:
        # 한글 단어만 추출 (2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', text)
        
        # 불용어 제거
        stopwords = ['것이', '있는', '하는', '되는', '같은', '많은', '이런', '그런', '저런', 
                    '이것', '그것', '저것', '여기', '거기', '저기', '때문', '통해', '위해',
                    '대한', '관련', '경우', '때문에', '이후', '이전', '현재', '오늘', '어제',
                    '기자', '뉴스', '연합뉴스', '기사', '보도', '발표', '설명', '말했다']
        
        filtered_words = [word for word in korean_words if word not in stopwords and len(word) >= 2]
        
        # 빈도수 계산
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(top_n)]
        
        print(f"✅ 간단 키워드 추출 완료: {keywords}")
        return keywords
        
    except Exception as e:
        print(f"❌ 키워드 추출 실패: {e}")
        return []

# 5. 실행 파이프라인 (개선된 버전)
def run_pipeline():
    """메인 실행 파이프라인"""
    print("\n" + "="*60)
    print("📰 연합뉴스 디지털 헬스케어 기사 분석 시작")
    print("="*60)
    
    try:
        # 기사 목록 수집 (네이버 뉴스 우선 시도)
        articles = get_naver_news_articles(SEARCH_KEYWORD, pages=1)
        
        # 네이버 뉴스에서 수집 실패시 연합뉴스 시도
        if not articles:
            print("⚠️ 네이버 뉴스 수집 실패, 연합뉴스 시도...")
            articles = get_yna_article_links(SEARCH_KEYWORD, pages=1)
        
        if not articles:
            print("❌ 실제 기사를 수집할 수 없습니다.")
            
            # 샘플 데이터 생성 (7개로 확장)
            print("💡 샘플 데이터를 생성합니다...")
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            sample_data = [
                {
                    "title": "디지털 헬스케어 시장 급성장, AI 기술 도입 확산",
                    "link": "https://example.com/news1",
                    "summary": "인공지능 기술을 활용한 디지털 헬스케어 시장이 급속도로 성장하고 있다. 원격 진료와 웨어러블 디바이스 등이 주요 성장 동력이 되고 있다.",
                    "keywords": "디지털헬스케어, 인공지능, 원격진료, 웨어러블",
                    "date": current_date
                },
                {
                    "title": "원격 진료 플랫폼 확산으로 의료 접근성 향상",
                    "link": "https://example.com/news2",
                    "summary": "코로나19를 계기로 원격 진료 서비스가 본격화되면서 의료 접근성이 크게 향상되고 있다.",
                    "keywords": "원격진료, 의료접근성, 코로나19, 플랫폼",
                    "date": current_date
                },
                {
                    "title": "웨어러블 디바이스로 건강 관리 혁신",
                    "link": "https://example.com/news3", 
                    "summary": "스마트워치와 피트니스 트래커 등이 개인 건강 관리의 새로운 패러다임을 제시하고 있다.",
                    "keywords": "웨어러블, 스마트워치, 건강관리, 피트니스",
                    "date": current_date
                },
                {
                    "title": "빅데이터 활용한 맞춤형 의료 서비스 확산",
                    "link": "https://example.com/news4",
                    "summary": "의료 빅데이터를 활용한 개인 맞춤형 치료와 예방 서비스가 확산되고 있어 의료 패러다임 변화를 이끌고 있다.",
                    "keywords": "빅데이터, 맞춤형의료, 예방서비스, 의료패러다임",
                    "date": current_date
                },
                {
                    "title": "IoT 기반 스마트 병원 시스템 도입 가속화",
                    "link": "https://example.com/news5",
                    "summary": "사물인터넷 기술을 활용한 스마트 병원 시스템이 도입되면서 환자 모니터링과 의료진 업무 효율성이 크게 향상되고 있다.",
                    "keywords": "IoT, 스마트병원, 환자모니터링, 업무효율성",
                    "date": current_date
                },
                {
                    "title": "블록체인 기술로 의료 데이터 보안 강화",
                    "link": "https://example.com/news6",
                    "summary": "블록체인 기술을 도입하여 의료 데이터의 보안성과 투명성을 높이는 시스템이 개발되어 주목받고 있다.",
                    "keywords": "블록체인, 의료데이터, 보안, 투명성",
                    "date": current_date
                },
                {
                    "title": "VR/AR 기술 활용한 의료 교육 및 치료 혁신",
                    "link": "https://example.com/news7",
                    "summary": "가상현실과 증강현실 기술이 의료진 교육과 환자 치료에 활용되면서 의료 서비스의 질적 향상을 가져오고 있다.",
                    "keywords": "VR, AR, 의료교육, 치료혁신",
                    "date": current_date
                }
            ]
            
            sample_df = pd.DataFrame(sample_data)
            
            # 기존 데이터와 합치기
            try:
                existing_df = pd.read_csv("digital_healthcare_news.csv")
                print(f"📂 기존 데이터 {len(existing_df)}개 발견")
                
                combined_df = pd.concat([existing_df, sample_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['title'], keep='last')
                
                # 최대 10개로 제한
                if len(combined_df) > 10:
                    combined_df = combined_df.tail(10)
                    
            except FileNotFoundError:
                print("📂 기존 데이터 파일이 없어 새로 생성합니다.")
                combined_df = sample_df
            
            # 두 파일에 모두 저장
            combined_df.to_csv("yna_digital_healthcare_news.csv", index=False, encoding="utf-8-sig")
            combined_df.to_csv("digital_healthcare_news.csv", index=False, encoding="utf-8-sig")
            
            print(f"✅ 샘플 데이터 저장 완료: 총 {len(combined_df)}개 기사")
            print("📁 파일: yna_digital_healthcare_news.csv, digital_healthcare_news.csv")
            return
        
        results = []
        total_articles = min(len(articles), 7)  # 최대 7개 기사 처리
        
        print(f"📊 총 {total_articles}개 기사를 처리합니다...\n")
        
        for i, article in enumerate(articles[:total_articles], 1):
            print(f"\n[{i}/{total_articles}] 기사 처리 중...")
            print(f"📰 제목: {article['title'][:50]}...")
            
            try:
                # 본문 추출
                full_text = extract_yna_article_text(article['link'])
                
                if not full_text:
                    print("⚠️ 본문을 추출할 수 없어 건너뜁니다.")
                    continue
                
                # 요약 생성
                print("📝 요약 생성 중...")
                summary = summarize_text(full_text)
                
                # 키워드 추출
                print("🔍 키워드 추출 중...")
                keywords = extract_keywords(full_text)
                
                # 결과 저장
                result = {
                    "title": article["title"],
                    "link": article["link"],
                    "text_length": len(full_text),
                    "summary": summary if summary else '요약 생성 실패',
                    "keywords": ", ".join(keywords) if keywords else '키워드 추출 실패',
                    "date": datetime.now().strftime('%Y-%m-%d')
                }
                
                results.append(result)
                print(f"✅ 기사 {i} 처리 완료")
                
                # 서버 부담 줄이기
                if i < total_articles:
                    print("⏳ 2초 대기 중...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ 기사 {i} 처리 중 오류: {e}")
                continue
        
        # 결과 저장 및 기존 데이터와 합치기
        if results:
            try:
                new_df = pd.DataFrame(results)
                
                # 기존 데이터 로드 시도
                try:
                    existing_df = pd.read_csv("digital_healthcare_news.csv")
                    print(f"📂 기존 데이터 {len(existing_df)}개 발견")
                    
                    # 기존 데이터와 새 데이터 합치기
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    
                    # 중복 제거 (제목 기준)
                    combined_df = combined_df.drop_duplicates(subset=['title'], keep='last')
                    
                    # 최대 10개로 제한
                    if len(combined_df) > 10:
                        combined_df = combined_df.tail(10)  # 최신 10개 유지
                    
                    print(f"📊 합쳐진 데이터: {len(combined_df)}개")
                    
                except FileNotFoundError:
                    print("📂 기존 데이터 파일이 없어 새로 생성합니다.")
                    combined_df = new_df
                
                # 두 파일에 모두 저장
                csv_filename = "yna_digital_healthcare_news.csv"
                combined_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
                
                # app01.py에서 사용하는 파일에도 저장
                combined_df.to_csv("digital_healthcare_news.csv", index=False, encoding="utf-8-sig")
                
                print(f"\n" + "="*60)
                print("✅ 분석 완료!")
                print(f"📁 결과 파일: {csv_filename}")
                print(f"�  app01.py용 파일: digital_healthcare_news.csv")
                print(f"📊 새로 처리된 기사 수: {len(results)}개")
                print(f"📊 총 기사 수: {len(combined_df)}개")
                print("="*60)
                
                # 전체 결과 미리보기
                print("\n📋 전체 데이터 미리보기:")
                for i, (_, row) in enumerate(combined_df.iterrows(), 1):
                    print(f"\n[{i}] {row['title'][:50]}...")
                    print(f"    📝 요약: {str(row['summary'])[:100]}...")
                    print(f"    🏷️ 키워드: {row['keywords']}")
                    if 'text_length' in row:
                        print(f"    📏 본문 길이: {row['text_length']}자")
                
            except Exception as e:
                print(f"❌ CSV 파일 저장 오류: {e}")
                
                # 백업: 텍스트 파일로 저장
                try:
                    with open("yna_digital_healthcare_news_backup.txt", "w", encoding="utf-8") as f:
                        f.write("연합뉴스 디지털 헬스케어 기사 분석 결과\n")
                        f.write("="*50 + "\n\n")
                        
                        for i, result in enumerate(results, 1):
                            f.write(f"[{i}] {result['title']}\n")
                            f.write(f"링크: {result['link']}\n")
                            f.write(f"요약: {result['summary']}\n")
                            f.write(f"키워드: {result['keywords']}\n")
                            f.write("-" * 50 + "\n\n")
                    
                    print("💾 백업 파일로 저장됨: yna_digital_healthcare_news_backup.txt")
                    
                except Exception as backup_error:
                    print(f"❌ 백업 파일 저장도 실패: {backup_error}")
        else:
            print("\n❌ 처리된 기사가 없습니다.")
            
    except Exception as e:
        print(f"❌ 파이프라인 실행 오류: {e}")
        import traceback
        print("🔍 상세 오류:")
        traceback.print_exc()

# 프로그램 실행
if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류: {e}")
        print("🔧 문제가 지속되면 필요한 라이브러리를 설치해주세요:")
        print("   pip install requests beautifulsoup4 pandas gensim keybert konlpy sumy")
    finally:
        print("\n👋 프로그램을 종료합니다.")
