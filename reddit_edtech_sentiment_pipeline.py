"""
REDDIT MEGA SCRAPER
By: Hugo Sanchez
07/21/2025
===================================

Purpose
-------
Scrape discussion data from a set of education- and startup-related subreddits,
extract feedback about online learning / AI education tools, identify feature
wishlist items, and produce aggregated sentiment metrics for common edtech
products (Duolingo, Khan Academy, Coursera, Quizlet, etc.).

Key Capabilities
----------------
1. **Reddit Scrape (Posts + Comments)** using PRAW.
2. **Product Mention Detection** (configurable patterns & synonyms).
3. **Nitter + VADER Sentiment Analysis** (per text + aggregated by product).
4. **Feature Wishlist Extraction** via regex heuristics + keyword spotting.
5. **AI / Online Learning Feedback Mining** (keyword-filtered corpus summary).
6. **CSV & JSON Outputs** for downstream analysis / dashboards.
7. **CLI Interface** for basic parameterization (limits, time filter, output dir).

Subreddits Targeted (Note that if you provide an unreachable subreddit, the program will throw a HTTP 403 Error and continue running)
----------------------------------------------------
- r/edtech                 - r/education
- r/teachers               - r/teaching
- r/students               - r/teacherslounge
- r/onlinelearning         - r/gradschool
- r/homeschool             - r/askacademia
- r/college                - r/machinelearning
- r/learnprogramming       - r/artificial
- r/duolingo               - r/edtechbooks
- r/khanacademy            - r/languagetechnology
- r/quizlet                - r/chatgpt
- r/coursera               - r/getstudying
- r/SaaS                   - r/learning
- r/startups               - r/highereducation
- r/askteachers            - r/teachersofreddit
- r/educationreform        - r/openedtech
- r/elearning              - r/k12sysadmin
- r/edtechstartups         - r/collegeapps
- r/studentsuccess         - r/learningmath
- r/languagelearning       - r/educationaltechnology
- r/openeducation          - r/academicchatter
- r/teacherspayteachers    - r/adulteducation
- r/physicsstudents        - r/askstudents
- r/edtechnews             



Install Requirements
--------------------
This script is intentionally light on dependencies. Install with:

```bash
pip install praw pandas tqdm nltk scikit-learn
```

Then (first run only) download the VADER lexicon:

```bash
python - <<'EOF'
import nltk; nltk.download('vader_lexicon')
EOF
```

Reddit API Credentials
----------------------
Export env vars before running (recommended):

```bash
export REDDIT_CLIENT_ID="9h5VJ3uVfejgQL3-NGaL_g"
export REDDIT_CLIENT_SECRET="lA673vxCGRLa9LCVi1_vaJLmvIEtOA"
export REDDIT_USER_AGENT="rslashedtechscrapper"
```

Usage
-----

python reddit_edtech_sentiment_pipeline.py
  --limit 500 \
  --comments-limit 200 \
  --time-filter month \
  --output-dir ./reddit_edtech_out
```

Optional arguments include `--products-csv` to supply your own product list and
patterns; see `DEFAULT_PRODUCTS` below for schema.

Outputs (All optional; created if data available)
-------------------------------------------------
- `raw_posts.csv`            : One row per submission (basic metadata)
- `raw_comments.csv`         : One row per comment (flattened tree)
- `raw_texts.jsonl`          : Unified text blobs (post + top-level comment context)
- `product_sentiment.csv`    : Aggregated VADER stats per product (mean, counts)
- `feature_wishlist.csv`     : Extracted candidate feature requests (deduped + counts)
- `ai_feedback_phrases.csv`  : Frequent n-grams from AI-related corpus slice
- `ai_feedback_examples.csv` : Top positive / negative / neutral examples (sample)

Data Model Notes
----------------
- Posts + comments are kept in separate frames; a derived `texts` frame merges
  them for NLP.
- All times are UTC epoch seconds from Reddit, converted to ISO8601.
- Sentiment is computed on **cleaned text** (urls stripped, markdown removed-ish).
- Product detection is case-insensitive regex; product names can overlap—longer
  regexes are applied first.



To Run: (instantiates your client/secret and installs required nltk packages for sentiment compound reading)
-----------
- IMPORTANT Paste this into cmd:
python reddit_edtech_sentiment_pipeline.py --client-id (PASTE YOUR CLIENT ID HERE) --client-secret (PASTE YOUR CLIENT SECRET HERE)
- Will take upwards of 2 HOURS to run depending primarily on the post and comment limit args as well as the # of subreddits

"""


from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterable, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import praw
from praw.models import Submission, Comment

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# CONFIGURATION (Subreddits targeted, products, etc)
# ---------------------------------------------------------------------------
SUBREDDITS = [
    "edtech",
    "teachers",
    "students",
    "onlinelearning",
    "homeschool",
    "college",
    "learnprogramming",
    "duolingo",
    "khanacademy",
    "quizlet",
    "coursera",
    "SaaS",
    "startups",
    "education",
    "teaching",
    "teacherslounge",
    "gradschool",
    "askacademia",
    "machinelearning",
    "artificial",
    "edtechbooks",
    "languagetechnology",
    "chatgpt",
    "getstudying",
    "learning",
    "highereducation",
    "askteachers",
    "teachersofreddit",
    "educationreform",
    "openedtech",
    "elearning",
    "k12sysadmin",
    "edtechstartups",
    "collegeapps",
    "studentsuccess",
    "learningmath",
    "languagelearning",
    "educationaltechnology",
    "openeducation",
    "academicchatter",
    "teacherspayteachers",
    "adulteducation",
    "physicsstudents",
    "askstudents",
    "edtechnews"
]

# Product patterns: map canonical product key -> dict(name=display_name, patterns=[regex,...])
# NOTE: order matters when scanning text (longest / most specific first is better).
DEFAULT_PRODUCTS = [
    {
        "key": "duolingo",
        "name": "Duolingo",
        "patterns": [r"duo\s*lingo", r"duolingo"]
    },
    {
        "key": "khan_academy",
        "name": "Khan Academy",
        "patterns": [r"khan\s*academy", r"khanacademy"]
    },
    {
        "key": "coursera",
        "name": "Coursera",
        "patterns": [r"coursera"]
    },
    {
        "key": "quizlet",
        "name": "Quizlet",
        "patterns": [r"quizlet"]
    },
    {
        "key": "edx",
        "name": "edX",
        "patterns": [r"\bedx\b", r"ed\s*x(?!t)"]
    },
    {
        "key": "udemy",
        "name": "Udemy",
        "patterns": [r"udemy"]
    },
    {
        "key": "byju",
        "name": "BYJU'S",
        "patterns": [r"byju", r"byjus", r"byju's"]
    },
    {
        "key": "photomath",
        "name": "Photomath",
        "patterns": [r"photo\s*math", r"photomath"]
    },
    {
        "key": "ixl",
        "name": "IXL",
        "patterns": [r"\bixl\b"]
    },
    {
        "key": "chatgpt",
        "name": "ChatGPT",
        "patterns": [r"chat\s*gpt", r"chatgpt"]
    },
    {
        "key": "claude",
        "name": "Claude",
        "patterns": [r"claude"]
    },
    {
        "key": "perplexity",
        "name": "Perplexity",
        "patterns": [r"perplexity"]
    },
    {
        "key": "magic_school_ai",
        "name": "Magic School AI",
        "patterns": [r"magic\s*school\s*ai", r"magicschoolai"]
    },
    {
        "key": "brainly",
        "name": "Brainly",
        "patterns": [r"brainly"]
    },
    {
        "key": "chegg",
        "name": "Chegg",
        "patterns": [r"chegg"]
    },
    {
        "key": "course_hero",
        "name": "Course Hero",
        "patterns": [r"course\s*hero", r"coursehero"]
    },
    {
        "key": "gauthmath",
        "name": "Gauthmath",
        "patterns": [r"gauth\s*math", r"gauthmath"]
    },
    {
        "key": "opennote",
        "name": "Opennote",
        "patterns": [r"open\s*note", r"opennote"]
    },
    {
        "key": "chiron",
        "name": "Chiron",
        "patterns": [r"chiron"]
    },
    {
        "key": "youlearn",
        "name": "YouLearn",
        "patterns": [r"you\s*learn", r"youlearn"]
    },
    {
        "key": "miyagi_labs",
        "name": "Miyagi Labs",
        "patterns": [r"miyagi\s*labs?", r"miyagilabs?"]
    },
    {
        "key": "alice_tech",
        "name": "Alice.Tech",
        "patterns": [r"alice\s*\.?tech", r"alicetech"]
    },
    {
        "key": "excellence_learning",
        "name": "Excellence Learning",
        "patterns": [r"excellence\s*learning", r"excellencelearning"]
    },
    {
        "key": "revisiondojo",
        "name": "RevisionDojo",
        "patterns": [r"revision\s*dojo", r"revisiondojo"]
    },
    {
        "key": "educato_ai",
        "name": "Educato AI",
        "patterns": [r"educato\s*ai", r"educatoai"]
    },
    {
        "key": "mathos",
        "name": "Mathos",
        "patterns": [r"mathos"]
    },
    {
        "key": "shepherd",
        "name": "Shepherd",
        "patterns": [r"shepherd"]
    },
    {
        "key": "ultra",
        "name": "Ultra",
        "patterns": [r"\bultra\b"]
    },
    {
        "key": "mathdash",
        "name": "MathDash",
        "patterns": [r"math\s*dash", r"mathdash"]
    },
    {
        "key": "flint",
        "name": "Flint",
        "patterns": [r"flint"]
    },
    {
        "key": "studdy",
        "name": "Studdy",
        "patterns": [r"studdy"]
    },
    {
        "key": "univerbal",
        "name": "Univerbal",
        "patterns": [r"univerbal"]
    },
    {
        "key": "superkalam",
        "name": "SuperKalam",
        "patterns": [r"super\s*kalam", r"superkalam"]
    },
    {
        "key": "darsel",
        "name": "Darsel",
        "patterns": [r"darsel"]
    },
    {
        "key": "marathon_education",
        "name": "Marathon Education",
        "patterns": [r"marathon\s*education", r"marathoneducation"]
    },
    {
        "key": "toko",
        "name": "Toko",
        "patterns": [r"toko"]
    },
    {
        "key": "mysivi",
        "name": "MySivi",
        "patterns": [r"my\s*sivi", r"mysivi"]
    },
    {
        "key": "studystream",
        "name": "StudyStream",
        "patterns": [r"study\s*stream", r"studystream"]
    },
    {
        "key": "underleaf",
        "name": "Underleaf",
        "patterns": [r"under\s*leaf", r"underleaf"]
    },
]


# AI / Online Learning keywords (broad net; lower-case match)
AI_KEYWORDS = [
    
    "ai", "artificial intelligence", "machine learning", "ml", "gpt", "chatgpt",
    "claude", "llm", "ai tutor", "ai-powered", "ai tool", "ai assistant", "copilot",
    "genai", "gpt-4", "gpt4", "anthropic", "openai", "deepseek", "llama", "mistral",
    "tutor", "teaching assistant",

    # Expanded AI & LLM keywords
    "gpt-3", "gpt3", "gpt-4o", "gpt-5", "gemini", "bard", "palm", "falcon", "orca",
    "ai chatbot", "ai educator", "ai instructor", "ai teaching assistant",
    "personalized learning", "intelligent tutor", "virtual tutor", "digital tutor",
    "chatbot tutor", "ai coach", "adaptive learning", "edtech ai", "ai for education",

    # Tools & platforms
    "duolingo ai", "quizlet q-chat", "khanmigo", "copilot for education",
    "ai homework helper", "homework ai", "math ai tutor", "socratic", "photomath",
    "wolfram alpha", "caktus ai", "writesonic", "perplexity", "pi ai", "replika",
    "notion ai", "slack ai", "bing chat", "huggingface", "cohere", "stability ai",

    # Learning & teaching enhancements
    "automated grading", "essay grading ai", "smart content", "ai test prep",
    "ai curriculum", "learning analytics", "ai assessment", "virtual classroom ai",
    "voice tutor", "speech-to-text ai", "text-to-speech ai", "ai flashcards",
    "ai lesson planning", "content recommendation ai", "skill assessment ai",
    "gamified ai learning", "ai study partner",

    # General AI terms used in edtech
    "transformer models", "nlp", "language model", "large language model",
    "generative ai", "ai generation", "ai automation", "ai detection",
    "self-learning ai", "deep learning", "neural network", "autograder",
    "education llm", "ai note taker", "smart quiz generator", "quiz ai",
    "ai presentations", "lecture summarizer",
]

# Feature wishlist cue phrases (lower-case). We'll search for these and pull local spans.
WISHLIST_CUES = [
    # Original cues
    "i wish", "wish it", "would be nice", "would love", "needs to", "should have",
    "should add", "add a", "add an", "feature request", "missing", "lacking",
    "please add", "could you add", "need a way", "need an option", "need to be able",
    "it'd be great", "it would be great", "it would help if", "i want", "we want",

    # Expanded variations
    "i hope they add", "i hope it has", "should include", "could use", "want a feature",
    "wish there was", "i would like", "we would like", "hope to see", "please improve",
    "please support", "please implement", "please include", "request for feature",
    "feature suggestion", "feature idea", "could improve", "could add", "missing feature",
    "needs improvement", "add support for", "would appreciate", "please allow",
    "i’d appreciate", "i’d like to see", "i’d love to see", "it should have",
    "it’s lacking", "it’s missing", "needs upgrade", "better if it had",
    "important feature missing", "add functionality", "add capability", "add support",

    # More natural user requests
    "this app needs", "this site needs", "this tool needs", "desperately need",
    "must have feature", "essential feature missing", "must include", "could benefit from",
    "i’m looking for", "it’s crucial to have", "needs a fix", "improvement request",
    "please make it possible", "it’d be nice if", "i’d wish", "i’d hope",
    "add custom", "add new feature", "please enhance", "wish list item", "wishlist idea",
    "i recommend adding", "future feature", "next update should", "future update request",
    "add option to", "add ability to", "add feature to", "add toggle", "make it possible to",
    "allow us to", "allow me to", "i hope for", "we hope for",
]#Premise: the idea is that we can scrape user tool-requests to maybe innovate upon our tool better

# Regex patterns built at runtime from the cues above

# ---------------------------------------------------------------------------
# Utilities (i.e. creating )
# ---------------------------------------------------------------------------

def load_products_from_csv(path: str) -> List[Dict[str, Any]]:
    """Load product patterns from CSV.

    Expected columns: key,name,pattern1;pattern2;pattern3
    Example row:
        duolingo,Duolingo,duolingo;duo\s*lingo
    """
    df = pd.read_csv(path)
    products = []
    for _, row in df.iterrows():
        pats_raw = str(row.get("patterns") or row.get("pattern") or row.get("regex") or "").strip()
        if pats_raw:
            pats = [p.strip() for p in pats_raw.split(";") if p.strip()]
        else:
            pats = [re.escape(row['name'])]
        products.append({
            "key": row['key'],
            "name": row['name'],
            "patterns": pats,
        })
    return products


def compile_product_regex(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile regex for fast scanning. Returns dict key -> compiled pattern."""
    compiled = {}
    for p in products:
        # Combine patterns OR-wise, ensure case-insensitive
        pattern = "|".join(f"({pat})" for pat in p['patterns'])
        try:
            compiled[p['key']] = re.compile(pattern, flags=re.I)
        except re.error as e:
            print(f"[WARN] Bad regex for product {p['key']}: {e}; skipping", file=sys.stderr)
    return compiled


def compile_wishlist_regex(cues: List[str]) -> re.Pattern:
    # Escape cues that may contain regex chars; join with alternation.
    # We capture up to give or take 200 chars after cue to get phrase of question
    escaped = [re.escape(c) for c in cues]
    pattern = r"(?i)(?:" + "|".join(escaped) + r")(.*?)(?:[.!?\n]|$)"
    return re.compile(pattern)


URL_RE = re.compile(r"https?://\S+")
MARKDOWN_LINK_RE = re.compile(r"\[(.*?)\]\((.*?)\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str: #cleaning the text for non-reddit-post content that may be in the html string
    if not text:
        return ""
    # Remove markdown links but keep link text
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    # Strip raw URLs
    text = URL_RE.sub(" ", text)
    # Strip HTML-ish tags
    text = HTML_TAG_RE.sub(" ", text)
    # Decode HTML entities manually if needed
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Collapse whitespace
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Reddit Scrape
# ---------------------------------------------------------------------------

def get_reddit_client(client_id: str, client_secret: str, user_agent: str) -> praw.Reddit:
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
        ratelimit_seconds=5,
    )


SORT_FUNCS = {
    "hot": lambda sr, limit: sr.hot(limit=limit),
    "new": lambda sr, limit: sr.new(limit=limit),
    "top": lambda sr, limit: sr.top(limit=limit),
    "rising": lambda sr, limit: sr.rising(limit=limit),
}


def fetch_submissions(reddit: praw.Reddit, subreddit_name: str, limit: int, sort: str = "hot", time_filter: str = "all") -> List[Submission]:
    """Fetch submissions from a subreddit.

    For `top` we can pass `time_filter`; PRAW supports it via `.top(time_filter=...)`.
    """
    subreddit = reddit.subreddit(subreddit_name)
    submissions = []

    try:
        if sort == "top":
            iterator = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort == "new":
            iterator = subreddit.new(limit=limit)
        elif sort == "rising":
            iterator = subreddit.rising(limit=limit)
        else:
            iterator = subreddit.hot(limit=limit)
        for sub in iterator:
            submissions.append(sub)
    except Exception as e:
        print(f"[ERROR] Fetching {subreddit_name}: {e}", file=sys.stderr)
    return submissions


def fetch_comments_for_submission(submission: Submission, max_comments: int = 100) -> List[Comment]:
    """Fetch up to max_comments flattened comments for a submission."""
    try:
        submission.comments.replace_more(limit=None)
        all_comments = submission.comments.list()
    except Exception as e:
        print(f"[WARN] Could not expand comments for {submission.id}: {e}", file=sys.stderr)
        return []

    if max_comments and max_comments > 0:
        return all_comments[:max_comments]
    return all_comments


# ---------------------------------------------------------------------------
# Sentiment & Product Detection
# ---------------------------------------------------------------------------

def ensure_vader_downloaded():
    try:
        _ = SentimentIntensityAnalyzer()
    except Exception:
        nltk_download('vader_lexicon')


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    ensure_vader_downloaded()
    return SentimentIntensityAnalyzer()


def score_sentiment(analyzer: SentimentIntensityAnalyzer, text: str) -> Dict[str, float]:
    if not text:
        return {"pos": 0.0, "neu": 0.0, "neg": 0.0, "compound": 0.0}
    return analyzer.polarity_scores(text)


def detect_products_in_text(text: str, product_regex: Dict[str, re.Pattern]) -> List[str]:
    found = []
    if not text:
        return found
    for key, pattern in product_regex.items():
        if pattern.search(text):
            found.append(key)
    return found


# ---------------------------------------------------------------------------
# Feature Wishlist Extraction
# ---------------------------------------------------------------------------

def extract_wishlist_phrases(text: str, wishlist_re: re.Pattern, max_len: int = 200) -> List[str]:
    """Return candidate feature wishlist phrases following cue expressions.

    Example: "I wish Duolingo had offline mode" -> "Duolingo had offline mode"
    """
    if not text:
        return []
    out = []
    for m in wishlist_re.finditer(text):
        span = m.group(1) or ""
        span = span.strip(" :;,.!?")
        if span:
            span = span[:max_len]
            out.append(span)
    return out


# ---------------------------------------------------------------------------
# AI Feedback Mining
# ---------------------------------------------------------------------------

def contains_ai_keyword(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    lt = text.lower()
    return any(k in lt for k in keywords)


def top_ngrams(texts: Iterable[str], ngram_range=(1,2), top_k: int = 50, min_df: int = 2) -> List[Tuple[str, int]]:
    """Return top_k most frequent ngrams across corpus."""
    cv = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    X = cv.fit_transform(texts)
    freqs = X.sum(axis=0).A1
    vocab = cv.get_feature_names_out()
    pairs = list(zip(vocab, freqs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


# ---------------------------------------------------------------------------
# Data Assembly
# ---------------------------------------------------------------------------

def submission_to_dict(sub: Submission) -> Dict[str, Any]:
    return {
        "id": sub.id,
        "subreddit": str(sub.subreddit).lower(),
        "title": sub.title or "",
        "selftext": sub.selftext or "",
        "score": sub.score,
        "num_comments": sub.num_comments,
        "created_utc": sub.created_utc,
        "created_iso": dt.datetime.utcfromtimestamp(sub.created_utc).isoformat() + "Z",
        "permalink": f"https://www.reddit.com{sub.permalink}",
        "url": sub.url,
        "author": str(sub.author) if sub.author else None,
        "over_18": sub.over_18,
        "is_self": sub.is_self,
    }


def comment_to_dict(com: Comment, submission_id: str) -> Dict[str, Any]:
    body = com.body or ""
    return {
        "id": com.id,
        "submission_id": submission_id,
        "body": body,
        "score": com.score,
        "created_utc": getattr(com, 'created_utc', None),
        "created_iso": dt.datetime.utcfromtimestamp(getattr(com, 'created_utc', 0)).isoformat() + "Z" if getattr(com, 'created_utc', None) else None,
        "author": str(com.author) if com.author else None,
        "parent_id": com.parent_id,
        "link_id": com.link_id,
        "is_submitter": getattr(com, 'is_submitter', False),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    client_id: str,
    client_secret: str,
    user_agent: str,
    subreddits: List[str],
    limit: int,
    comments_limit: int,
    sort: str,
    time_filter: str,
    output_dir: str,
    products: List[Dict[str, Any]] = None,
    ai_keywords: List[str] = None,
    wishlist_cues: List[str] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    products = products or DEFAULT_PRODUCTS
    ai_keywords = ai_keywords or AI_KEYWORDS
    wishlist_cues = wishlist_cues or WISHLIST_CUES

    product_regex = compile_product_regex(products)
    wishlist_re = compile_wishlist_regex(wishlist_cues)

    reddit = get_reddit_client(client_id, client_secret, user_agent)

    # Collect submissions
    all_submissions: List[Dict[str, Any]] = []
    raw_submission_objs: List[Submission] = []

    print(f"[INFO] Fetching submissions from {len(subreddits)} subreddits...")
    for sr in subreddits:
        subs = fetch_submissions(reddit, sr, limit=limit, sort=sort, time_filter=time_filter)
        print(f"  - {sr}: {len(subs)} submissions")
        raw_submission_objs.extend(subs)
        for sub in subs:
            all_submissions.append(submission_to_dict(sub))

    posts_df = pd.DataFrame(all_submissions)
    posts_path = os.path.join(output_dir, "raw_posts.csv")
    posts_df.to_csv(posts_path, index=False)
    print(f"[WRITE] {posts_path} ({len(posts_df)} rows)")

    # Collect comments
    all_comments: List[Dict[str, Any]] = []
    print("[INFO] Fetching comments (this may take a while)...")
    for sub in tqdm(raw_submission_objs, desc="comments"):
        comments = fetch_comments_for_submission(sub, max_comments=comments_limit)
        for com in comments:
            all_comments.append(comment_to_dict(com, submission_id=sub.id))
        # Light rate-limit courtesy pause
        time.sleep(0.5)

    comments_df = pd.DataFrame(all_comments)
    comments_path = os.path.join(output_dir, "raw_comments.csv")
    comments_df.to_csv(comments_path, index=False)
    print(f"[WRITE] {comments_path} ({len(comments_df)} rows)")

    # Assemble unified text corpus (post + comment body) ----------------------------------
    texts_records = []

    # Posts as texts
    for _, row in posts_df.iterrows():
        text_blob = (row['title'] or "") + "\n" + (row['selftext'] or "")
        texts_records.append({
            "type": "post",
            "subreddit": row['subreddit'],
            "submission_id": row['id'],
            "comment_id": None,
            "orig_text": text_blob,
        })

    # Comments as texts
    for _, row in comments_df.iterrows():
        texts_records.append({
            "type": "comment",
            "subreddit": None,  # we'll fill from posts later
            "submission_id": row['submission_id'],
            "comment_id": row['id'],
            "orig_text": row['body'],
        })

    texts_df = pd.DataFrame(texts_records)

    # Fill subreddit for comments by joining posts_df
    texts_df = texts_df.merge(
        posts_df[['id', 'subreddit']],
        left_on='submission_id', right_on='id', how='left', suffixes=('', '_post')
    )
    texts_df['subreddit'] = texts_df['subreddit'].fillna(texts_df['subreddit_post'])
    texts_df.drop(columns=['id', 'subreddit_post'], inplace=True)

    # Clean text
    texts_df['clean_text'] = texts_df['orig_text'].apply(clean_text)

    # Sentiment ---------------------------------------------------------------------------
    analyzer = get_sentiment_analyzer()
    sent_scores = texts_df['clean_text'].apply(lambda t: score_sentiment(analyzer, t))
    sent_df = pd.DataFrame(list(sent_scores))
    texts_df = pd.concat([texts_df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)

    # Product mentions -------------------------------------------------------------------
    texts_df['products'] = texts_df['clean_text'].apply(lambda t: detect_products_in_text(t, product_regex))

    # Feature wishlist phrases ------------------------------------------------------------
    texts_df['wishlist_phrases'] = texts_df['clean_text'].apply(lambda t: extract_wishlist_phrases(t, wishlist_re))

    # AI keyword filter ------------------------------------------------------------------
    texts_df['ai_related'] = texts_df['clean_text'].apply(lambda t: contains_ai_keyword(t, ai_keywords))

    # Write raw texts JSONL ---------------------------------------------------------------
    texts_path = os.path.join(output_dir, "raw_texts.jsonl")
    with open(texts_path, 'w', encoding='utf-8') as f:
        for _, row in texts_df.iterrows():
            rec = row.to_dict()
            # lists not JSON serializable by default -> keep as is; json dumps ok
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[WRITE] {texts_path} ({len(texts_df)} rows)")

    # -----------------------------------------------------------------------
    # Aggregation: Product Sentiment
    # -----------------------------------------------------------------------
    prod_rows = []
    for _, row in texts_df.iterrows():
        prods = row['products']
        if not prods:
            continue
        for p in prods:
            prod_rows.append({
                "product_key": p,
                "subreddit": row['subreddit'],
                "type": row['type'],
                "submission_id": row['submission_id'],
                "comment_id": row['comment_id'],
                "text": row['clean_text'],
                "pos": row['pos'],
                "neu": row['neu'],
                "neg": row['neg'],
                "compound": row['compound'],
            })
    prod_df = pd.DataFrame(prod_rows)

    if not prod_df.empty:
        agg = prod_df.groupby('product_key').agg(
            n_mentions=('product_key', 'size'),
            compound_mean=('compound', 'mean'),
            compound_median=('compound', 'median'),
            pos_mean=('pos', 'mean'),
            neg_mean=('neg', 'mean'),
        ).reset_index()

        # classify share pos/neg by thresholds
        agg_detail = []
        for p, grp in prod_df.groupby('product_key'):
            pos_n = (grp['compound'] >= 0.05).sum()
            neg_n = (grp['compound'] <= -0.05).sum()
            neu_n = len(grp) - pos_n - neg_n
            agg_detail.append({
                'product_key': p,
                'n_pos': pos_n,
                'n_neg': neg_n,
                'n_neu': neu_n,
            })
        agg_detail_df = pd.DataFrame(agg_detail)
        prod_summary_df = agg.merge(agg_detail_df, on='product_key', how='left')

        # attach product display names
        key2name = {p['key']: p['name'] for p in products}
        prod_summary_df['product_name'] = prod_summary_df['product_key'].map(key2name)

        prod_summary_path = os.path.join(output_dir, "product_sentiment.csv")
        prod_summary_df.to_csv(prod_summary_path, index=False)
        print(f"[WRITE] {prod_summary_path} ({len(prod_summary_df)} rows)")

        # sample top positive/negative examples per product (up to 5 each)
        examples_rows = []
        for p, grp in prod_df.groupby('product_key'):
            top_pos = grp.sort_values('compound', ascending=False).head(5)
            top_neg = grp.sort_values('compound').head(5)
            for _, r2 in top_pos.iterrows():
                examples_rows.append({"product_key": p, "sentiment": "pos", "compound": r2['compound'], "text": r2['text'][:500]})
            for _, r2 in top_neg.iterrows():
                examples_rows.append({"product_key": p, "sentiment": "neg", "compound": r2['compound'], "text": r2['text'][:500]})
        examples_df = pd.DataFrame(examples_rows)
        examples_path = os.path.join(output_dir, "product_sentiment_examples.csv")
        examples_df.to_csv(examples_path, index=False)
        print(f"[WRITE] {examples_path} ({len(examples_df)} rows)")
    else:
        print("[INFO] No product mentions detected; skipping product sentiment outputs.")

    # -----------------------------------------------------------------------
    # Aggregation: Feature Wishlist
    # -----------------------------------------------------------------------
    wl_rows = []
    for _, row in texts_df.iterrows():
        phrases = row['wishlist_phrases']
        if not phrases:
            continue
        for ph in phrases:
            # simple normalization: lowercase, strip punctuation
            norm = re.sub(r"[\s\-_/]+", " ", ph.lower()).strip()
            wl_rows.append({
                "phrase": ph,
                "norm": norm,
                "subreddit": row['subreddit'],
                "products": ",".join(row['products']) if row['products'] else "",
                "text": row['clean_text'][:500],
            })

    wl_df = pd.DataFrame(wl_rows)
    if not wl_df.empty:
        wl_agg = wl_df.groupby('norm').agg(
            n=('norm', 'size'),
            example=('phrase', 'first'),
            subreddits=('subreddit', lambda x: ','.join(sorted(set(filter(None, x))))),
            products=('products', lambda x: ','.join(sorted({p for cell in x for p in cell.split(',') if p}))),
        ).reset_index(drop=False).sort_values('n', ascending=False)

        wl_path = os.path.join(output_dir, "feature_wishlist.csv")
        wl_agg.to_csv(wl_path, index=False)
        print(f"[WRITE] {wl_path} ({len(wl_agg)} rows)")
    else:
        print("[INFO] No wishlist phrases detected.")

    # -----------------------------------------------------------------------
    # Aggregation: AI Feedback Slice
    # -----------------------------------------------------------------------
    ai_df = texts_df[texts_df['ai_related'] & texts_df['clean_text'].str.len().gt(0)].copy()
    if not ai_df.empty:
        ai_texts = ai_df['clean_text'].tolist()
        ngram_pairs = top_ngrams(ai_texts, ngram_range=(1,2), top_k=100, min_df=2)
        ai_ngrams_df = pd.DataFrame(ngram_pairs, columns=['ngram', 'freq'])
        ai_ngrams_path = os.path.join(output_dir, "ai_feedback_phrases.csv")
        ai_ngrams_df.to_csv(ai_ngrams_path, index=False)
        print(f"[WRITE] {ai_ngrams_path} ({len(ai_ngrams_df)} rows)")

        # sample examples by sentiment
        ai_examples_rows = []
        # pos
        pos_ex = ai_df.sort_values('compound', ascending=False).head(20)
        for _, r in pos_ex.iterrows():
            ai_examples_rows.append({"sentiment": "pos", "compound": r['compound'], "text": r['clean_text'][:500]})
        # neg
        neg_ex = ai_df.sort_values('compound').head(20)
        for _, r in neg_ex.iterrows():
            ai_examples_rows.append({"sentiment": "neg", "compound": r['compound'], "text": r['clean_text'][:500]})
        # neutral sample
        neu_ex = ai_df[(ai_df['compound'].abs() < 0.05)].sample(min(20, len(ai_df[(ai_df['compound'].abs() < 0.05)])), random_state=0) if not ai_df[(ai_df['compound'].abs() < 0.05)].empty else pd.DataFrame()
        for _, r in neu_ex.iterrows():
            ai_examples_rows.append({"sentiment": "neu", "compound": r['compound'], "text": r['clean_text'][:500]})
        ai_examples_df = pd.DataFrame(ai_examples_rows)
        ai_examples_path = os.path.join(output_dir, "ai_feedback_examples.csv")
        ai_examples_df.to_csv(ai_examples_path, index=False)
        print(f"[WRITE] {ai_examples_path} ({len(ai_examples_df)} rows)")
    else:
        print("[INFO] No AI-related texts detected.")

    print("[DONE] Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

#IMPORTANT: These are your default values for all the args of the tool
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scrape Reddit edtech subs and analyze sentiment + feature requests.")
    p.add_argument("--client-id", default=os.getenv("REDDIT_CLIENT_ID"), help="Reddit API client ID (or env REDDIT_CLIENT_ID)")
    p.add_argument("--client-secret", default=os.getenv("REDDIT_CLIENT_SECRET"), help="Reddit API client secret (or env REDDIT_CLIENT_SECRET)")
    p.add_argument("--user-agent", default=os.getenv("REDDIT_USER_AGENT", "edtech-scraper/0.1"), help="User agent string")
    p.add_argument("--limit", type=int, default=250, help="Posts per subreddit")
    p.add_argument("--comments-limit", type=int, default=150, help="Max comments per submission (approx; after full expand we slice)")
    p.add_argument("--sort", choices=list(SORT_FUNCS.keys()), default="hot", help="Submission listing sort")
    p.add_argument("--time-filter", choices=["all", "day", "week", "month", "year"], default="month", help="Time filter for top sort")
    p.add_argument("--output-dir", default="./reddit_edtech_out", help="Directory for outputs")
    p.add_argument("--products-csv", default=None, help="Optional CSV of product patterns")
    p.add_argument("--subreddits", default=None, help="Comma-separated subreddit override")
    return p.parse_args(argv)


def main(argv=None): #Executes and instantiates the scraper
    args = parse_args(argv)

    if not args.client_id or not args.client_secret:
        print("[FATAL] Reddit API credentials missing. Use --client-id/--client-secret or env vars.", file=sys.stderr)
        sys.exit(1)

    subs = SUBREDDITS if args.subreddits is None else [s.strip() for s in args.subreddits.split(',') if s.strip()]

    products = DEFAULT_PRODUCTS
    if args.products_csv:
        products = load_products_from_csv(args.products_csv)

    run_pipeline(
        client_id=args.client_id,
        client_secret=args.client_secret,
        user_agent=args.user_agent,
        subreddits=subs,
        limit=args.limit,
        comments_limit=args.comments_limit,
        sort=args.sort,
        time_filter=args.time_filter,
        output_dir=args.output_dir,
        products=products,
    )


if __name__ == "__main__":
    main()
