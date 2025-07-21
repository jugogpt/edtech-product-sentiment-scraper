# Reddit EdTech Sentiment Pipeline

**Author**: Hugo Sanchez\
**Date**: July 21, 2025

## Overview

The Reddit EdTech Sentiment Pipeline is a Python-based tool designed to scrape and analyze discussions from education and startup-related subreddits. It extracts feedback about online learning and AI education tools, identifies feature wishlist items, and produces aggregated sentiment metrics for popular edtech products such as Duolingo, Khan Academy, Coursera, Quizlet, and others. The pipeline leverages the PRAW library for Reddit scraping, VADER for sentiment analysis, and regex-based heuristics for feature extraction.

## Key Features

- **Reddit Scraping**: Collects posts and comments from specified subreddits using PRAW.
- **Product Mention Detection**: Identifies mentions of edtech products using configurable regex patterns and synonyms.
- **Sentiment Analysis**: Computes sentiment scores for text using VADER, aggregated by product.
- **Feature Wishlist Extraction**: Extracts user-suggested features via regex heuristics and keyword spotting.
- **AI/Online Learning Feedback**: Mines feedback related to AI and online learning using keyword filtering.
- **Output Formats**: Generates CSV and JSONL files for raw data, sentiment metrics, feature requests, and AI feedback.
- **CLI Interface**: Supports parameterization for post/comment limits, time filters, and output directories.

## Targeted Subreddits

The pipeline targets a comprehensive list of subreddits relevant to education, edtech, and startups, including but not limited to:

- r/edtech
- r/education
- r/teachers
- r/students
- r/onlinelearning
- r/duolingo
- r/khanacademy
- r/coursera
- r/quizlet
- r/startups
- r/machinelearning
- r/artificial
- ... (and many more; see full list in the code's `SUBREDDITS` configuration)

**Note**: If a subreddit is unreachable, the program logs an HTTP 403 error and continues processing other subreddits.

## Installation

### Prerequisites

The script requires minimal dependencies. Install them using:

```bash
pip install praw pandas tqdm nltk scikit-learn
```

On the first run, download the VADER lexicon for sentiment analysis:

```bash
python - <<'EOF'
import nltk; nltk.download('vader_lexicon')
EOF
```

Additionally, the script uses NLTK's `punkt` tokenizer, which is automatically downloaded during execution.

### Reddit API Credentials

To access Reddit's API, you need to set up API credentials. Export the following environment variables (recommended):

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="your_user_agent"
```

Alternatively, pass the credentials via command-line arguments (see Usage below).

## Usage

Run the pipeline with the following command, replacing `your_client_id` and `your_client_secret` with your Reddit API credentials:

```bash
python reddit_edtech_sentiment_pipeline.py \
  --client-id your_client_id \
  --client-secret your_client_secret \
  --limit 500 \
  --comments-limit 200 \
  --time-filter month \
  --output-dir ./reddit_edtech_out
```

### Command-Line Arguments

- `--client-id`: Reddit API client ID (or use env var `REDDIT_CLIENT_ID`).
- `--client-secret`: Reddit API client secret (or use env var `REDDIT_CLIENT_SECRET`).
- `--user-agent`: User agent string (default: `edtech-scraper/0.1`).
- `--limit`: Number of posts to scrape per subreddit (default: 250).
- `--comments-limit`: Maximum comments per submission (default: 150).
- `--sort`: Submission sorting method (`hot`, `new`, `top`, `rising`; default: `hot`).
- `--time-filter`: Time filter for `top` sort (`all`, `day`, `week`, `month`, `year`; default: `month`).
- `--output-dir`: Directory for output files (default: `./reddit_edtech_out`).
- `--products-csv`: Optional CSV file with custom product patterns (see `DEFAULT_PRODUCTS` in code for schema).
- `--subreddits`: Optional comma-separated list of subreddits to override defaults.

**Note**: The pipeline may take upwards of 2 hours to run, depending on the number of subreddits, post limit, and comment limit.

## Outputs

The pipeline generates the following output files in the specified `--output-dir` (if data is available):

- **raw_posts.csv**: Metadata for each Reddit submission (e.g., title, score, subreddit, timestamp).
- **raw_comments.csv**: Flattened comment data with submission IDs and metadata.
- **raw_texts.jsonl**: Unified text corpus combining post titles, selftext, and comments.
- **product_sentiment.csv**: Aggregated sentiment metrics (mean, median, counts) per product.
- **feature_wishlist.csv**: Deduplicated feature requests with frequency and associated products/subreddits.
- **ai_feedback_phrases.csv**: Frequent n-grams from AI-related text corpus.
- **ai_feedback_examples.csv**: Sampled positive, negative, and neutral examples from AI-related texts.

## Data Model

- **Posts and Comments**: Stored in separate DataFrames, with a merged `texts` DataFrame for NLP tasks.
- **Timestamps**: Reddit's UTC epoch seconds are converted to ISO8601 format.
- **Sentiment**: Computed on cleaned text (URLs and markdown removed) using VADER's compound, positive, negative, and neutral scores.
- **Product Detection**: Uses case-insensitive regex patterns; longer patterns are prioritized to avoid overlap.
- **Feature Wishlist**: Extracts phrases following cue expressions (e.g., "I wish", "should add") with a 200-character limit.
- **AI Feedback**: Filters texts containing AI-related keywords and extracts top n-grams for analysis.

## Notes

- The pipeline is designed to handle large datasets but may encounter rate limits from Reddit's API. A courtesy pause is included to mitigate this.
- Product detection and feature extraction rely on regex patterns, which can be customized via the `--products-csv` argument.
- Sentiment analysis uses VADER, which is tuned for social media text but may not capture nuanced sentiment perfectly.
- The pipeline skips invalid regex patterns and logs errors for unreachable subreddits or other issues.
- Output files are only generated if relevant data (e.g., product mentions, wishlist phrases) is detected.

## License

This project is MIT licensed and provided as-is for educational purposes. Ensure compliance with Reddit's API terms and MIT license conditions when using this tool.
