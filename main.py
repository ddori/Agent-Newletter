"""
Self-Healing Agent Research Newsletter
arxiv 논문 수집 → Gemini API 요약 → Gmail 발송 자동화 파이프라인
"""

import os
import sys
import logging
import smtplib
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL") or GMAIL_ADDRESS
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

KEYWORDS = [
    "self-healing agent",
    "automated program repair",
    "self-healing network",
    "RAN self-optimization",
    "autonomous fault recovery",
    "self-adaptive robotics",
]

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS_PER_KEYWORD = 30
ARXIV_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------
@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str
    published: str
    arxiv_url: str
    pdf_url: str
    categories: list[str]
    summary_ko: str = ""
    relevance: str = ""


# ---------------------------------------------------------------------------
# Phase 1: Collect papers from arxiv
# ---------------------------------------------------------------------------
def _compute_days_back() -> int:
    """월요일이면 3일(주말 포함), 그 외 2일."""
    today = datetime.now(timezone.utc)
    return 3 if today.weekday() == 0 else 2


def _parse_arxiv_entry(entry: ET.Element) -> Paper:
    """arxiv Atom feed <entry>를 Paper 객체로 변환."""
    title = entry.findtext("atom:title", "", ARXIV_NAMESPACE).strip().replace("\n", " ")
    abstract = entry.findtext("atom:summary", "", ARXIV_NAMESPACE).strip().replace("\n", " ")
    published = entry.findtext("atom:published", "", ARXIV_NAMESPACE)[:10]

    authors = [
        name.text.strip()
        for name in entry.findall("atom:author/atom:name", ARXIV_NAMESPACE)
        if name.text
    ]

    arxiv_url = ""
    pdf_url = ""
    for link in entry.findall("atom:link", ARXIV_NAMESPACE):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
        elif link.get("rel") == "alternate":
            arxiv_url = link.get("href", "")

    categories = [
        cat.get("term", "")
        for cat in entry.findall("atom:category", ARXIV_NAMESPACE)
    ]

    return Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        published=published,
        arxiv_url=arxiv_url,
        pdf_url=pdf_url,
        categories=categories,
    )


def fetch_papers() -> list[Paper]:
    """키워드별로 arxiv API를 호출하고, 날짜 필터링 + 중복 제거."""
    days_back = _compute_days_back()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    seen_ids: dict[str, Paper] = {}

    for keyword in KEYWORDS:
        query = f'all:"{keyword}"'
        params = urllib.parse.urlencode({
            "search_query": query,
            "start": 0,
            "max_results": ARXIV_MAX_RESULTS_PER_KEYWORD,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{ARXIV_API_URL}?{params}"

        for attempt in range(3):
            try:
                logger.info(f"Fetching arxiv: {keyword} (attempt {attempt+1})")
                req = urllib.request.Request(url, headers={"User-Agent": "AgentNewsletter/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()

                root = ET.fromstring(data)
                entries = root.findall("atom:entry", ARXIV_NAMESPACE)
                logger.info(f"  Found {len(entries)} entries for '{keyword}'")

                for entry in entries:
                    paper = _parse_arxiv_entry(entry)
                    if paper.published < cutoff_str:
                        continue
                    arxiv_id = paper.arxiv_url.rstrip("/").split("/")[-1]
                    if arxiv_id not in seen_ids:
                        seen_ids[arxiv_id] = paper

                break  # 성공하면 재시도 루프 탈출
            except Exception as e:
                logger.warning(f"  Failed to fetch '{keyword}' (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))  # 5초, 10초 대기 후 재시도

        # arxiv API rate limit: 5초 간격
        time.sleep(5)

    papers = list(seen_ids.values())
    logger.info(f"Total unique papers after dedup: {len(papers)}")
    return papers


# ---------------------------------------------------------------------------
# Phase 2: Summarize with Gemini API
# ---------------------------------------------------------------------------
def _call_gemini(prompt: str) -> str:
    """Gemini REST API 직접 호출. 429시 재시도."""
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
    }).encode("utf-8")

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            return result["candidates"][0]["content"]["parts"][0]["text"]

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.warning(f"  Gemini attempt {attempt+1}: HTTP {e.code} - {error_body[:200]}")
            if attempt < 2:
                wait = 15 * (attempt + 1)
                logger.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
        except Exception as e:
            logger.warning(f"  Gemini attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(15 * (attempt + 1))

    return ""


def _summarize_one(paper: Paper) -> None:
    """단일 논문을 Gemini로 요약하고 Paper 객체에 저장."""
    prompt = f"""다음 논문을 분석해주세요.

제목: {paper.title}
저자: {', '.join(paper.authors[:5])}
초록: {paper.abstract}

다음 두 가지를 한국어로 작성해주세요:

1. **요약** (3-4문장): 이 논문의 핵심 내용, 방법론, 주요 결과를 요약해주세요.

2. **Agent-RAN 관련성**: 이 연구가 자가치유 에이전트(Self-healing Agent), RAN 자가최적화, \
자율 장애 복구, 또는 자가적응 시스템 관점에서 어떤 관련성이 있는지 1-2문장으로 평가해주세요.

반드시 아래 형식으로 응답해주세요:
[요약]
(요약 내용)

[관련성]
(관련성 평가)"""

    text = _call_gemini(prompt)
    if not text:
        paper.summary_ko = "(요약 생성 실패)"
        paper.relevance = "(관련성 평가 실패)"
        return

    if "[요약]" in text and "[관련성]" in text:
        parts = text.split("[관련성]")
        paper.summary_ko = parts[0].replace("[요약]", "").strip()
        paper.relevance = parts[1].strip()
    else:
        paper.summary_ko = text.strip()
        paper.relevance = "(관련성 평가 없음)"


def summarize_papers(papers: list[Paper]) -> list[Paper]:
    """모든 논문을 Gemini API로 요약."""
    if not papers:
        return papers

    for i, paper in enumerate(papers):
        logger.info(f"Summarizing [{i+1}/{len(papers)}]: {paper.title[:60]}")
        _summarize_one(paper)

    return papers


# ---------------------------------------------------------------------------
# Phase 3: Format HTML newsletter
# ---------------------------------------------------------------------------
def _paper_html(paper: Paper) -> str:
    """단일 논문을 HTML 블록으로 변환."""
    authors_str = ", ".join(paper.authors[:4])
    if len(paper.authors) > 4:
        authors_str += " 외"

    categories_str = ", ".join(paper.categories[:3])

    return f"""
    <div style="margin-top:25px; border-left:4px solid #667eea; padding:15px;
                background:#fafbfc; border-radius:0 8px 8px 0;">
      <h2 style="margin:0 0 8px 0; font-size:17px; color:#2c3e50;">
        <a href="{paper.arxiv_url}" style="color:#2c3e50; text-decoration:none;">
          {paper.title}
        </a>
      </h2>
      <p style="margin:0 0 8px 0; font-size:13px; color:#7f8c8d;">
        {authors_str} | {paper.published} | {categories_str}
      </p>
      <div style="margin:10px 0; padding:12px; background:white; border-radius:6px; line-height:1.6;">
        <strong>요약:</strong><br/>{paper.summary_ko}
      </div>
      <div style="margin:10px 0; padding:12px; background:#eef2ff; border-radius:6px; line-height:1.6;">
        <strong>Agent-RAN 관련성:</strong><br/>{paper.relevance}
      </div>
      <p style="margin:8px 0 0 0; font-size:13px;">
        <a href="{paper.arxiv_url}" style="color:#667eea;">arXiv</a> |
        <a href="{paper.pdf_url}" style="color:#667eea;">PDF</a>
      </p>
    </div>"""


def format_newsletter(papers: list[Paper], date_str: str) -> str:
    """논문 리스트를 HTML 뉴스레터로 포맷."""
    if papers:
        papers_html = "\n".join(_paper_html(p) for p in papers)
    else:
        papers_html = """
    <div style="text-align:center; padding:40px; color:#999;">
      <p style="font-size:16px;">오늘은 관련 논문이 발견되지 않았습니다.</p>
      <p>내일 다시 확인해보겠습니다.</p>
    </div>"""

    keywords_str = ", ".join(KEYWORDS)

    return f"""<html>
<body style="font-family:'Apple SD Gothic Neo','Malgun Gothic',sans-serif;
             max-width:700px; margin:0 auto; padding:20px; color:#333;">

  <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
              padding:30px; border-radius:10px; color:white; text-align:center;">
    <h1 style="margin:0; font-size:22px;">Self-Healing Agent 연구 동향</h1>
    <p style="margin:10px 0 0 0; opacity:0.9;">{date_str} | 논문 {len(papers)}편</p>
  </div>

  <div style="margin-top:10px; padding:12px; background:#f8f9fa;
              border-radius:8px; font-size:13px; color:#666;">
    <strong>검색 키워드:</strong> {keywords_str}
  </div>

  {papers_html}

  <hr style="margin-top:30px; border:none; border-top:1px solid #eee;"/>
  <p style="font-size:11px; color:#999; text-align:center;">
    이 뉴스레터는 GitHub Actions에 의해 자동 생성되었습니다.<br/>
    Powered by arXiv API + Gemini API
  </p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Phase 4: Send email via Gmail SMTP
# ---------------------------------------------------------------------------
def send_email(html_body: str, date_str: str) -> None:
    """Gmail SMTP를 통해 뉴스레터 발송."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Agent Research] Self-Healing Agent 연구 동향 - {date_str}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())

    logger.info(f"Newsletter sent to {RECIPIENT_EMAIL}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info(f"=== Newsletter generation started for {date_str} ===")

    # 1. Collect
    papers = fetch_papers()

    # 2. Summarize
    papers = summarize_papers(papers)

    # 3. Format
    html = format_newsletter(papers, date_str)

    # 4. Send
    send_email(html, date_str)

    logger.info(f"=== Done. {len(papers)} papers processed ===")


if __name__ == "__main__":
    main()
