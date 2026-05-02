from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = PROJECT_ROOT / "outputs" / "moderation_queue" / "business_month_queue"
REVIEWS_PATH = PROJECT_ROOT / "outputs" / "moderation_queue" / "top_suspicious_reviews"
LABEL_PATH = PROJECT_ROOT / "data" / "labels" / "review_labels.csv"

TECHNICAL_SIGNAL_DEFINITIONS = {
    "review_length": "Number of characters in the review text.",
    "word_count": "Approximate number of words in the review.",
    "useful": "Number of users who marked the review helpful, if available.",
    "helpful_vote": "Number of users who marked the review helpful, if available.",
    "rating_deviation_from_product_average": "How far this rating is from the product's average rating.",
    "anomaly_score": "Supporting model score for statistical unusualness. It does not surface reviews by itself.",
    "fraud_like_score": "Reviewer-facing priority score for high-confidence review manipulation signals.",
    "synthetic_templated_language_score": "Weak prioritization signal for generic, templated, overly promotional, or low-specificity language. This is not an AI detector.",
    "verified_purchase": "Whether Amazon marked the review as a verified purchase, if available.",
    "reviewer_average_rating": "The reviewer's average rating across their reviews.",
    "reviewer_review_count": "Number of reviews observed for that reviewer.",
    "reviewer_extreme_rating_share": "Share of that reviewer's reviews that are 1-star or 5-star.",
    "reviewer_five_star_share": "Share of that reviewer's reviews that are 5-star.",
    "reviewer_one_star_share": "Share of that reviewer's reviews that are 1-star.",
    "uppercase_share": "Share of text characters that are uppercase.",
    "repeated_character_flag": "Whether the text contains repeated characters.",
    "generic_praise_flag": "Whether the text contains short generic praise.",
    "html_break_count": "Count of HTML line-break tags found in the review text.",
    "text_formatting_anomaly_flag": "Whether broken spacing or excessive line breaks were detected.",
    "sentence_count": "Approximate number of sentences.",
    "avg_sentence_length": "Approximate average number of words per sentence.",
    "sentence_length_variability": "Approximate variation in sentence lengths.",
    "first_person_flag": "Whether the review uses first-person language.",
    "concrete_detail_flag": "Whether the review includes concrete product-use details.",
    "fraud_signal_count": "Count of concrete manipulation-oriented signals present on the review.",
}

st.set_page_config(page_title="Marketplace Integrity Monitor", layout="wide")


def read_output_dir(path: Path) -> pd.DataFrame:
    csv_path = path / "data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            st.warning(f"Could not read output at {path}: {exc}")
            return pd.DataFrame()
    return pd.DataFrame()


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def clean_review_text(value: object) -> str:
    if is_missing(value):
        return ""
    text = str(value)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > 20 and text.count("\n") > len(text) * 0.35:
        text = text.replace("\n", " ")
    return text.strip()


def month_key(value: object) -> str | None:
    if is_missing(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def format_month(value: object) -> str:
    key = month_key(value)
    if not key:
        return "Unknown month"
    parsed = pd.to_datetime(key, errors="coerce")
    if pd.isna(parsed):
        return key
    return parsed.strftime("%b %Y")


def product_key(row: pd.Series) -> str | None:
    for column in ("asin", "product_id", "business_id"):
        if column in row and not is_missing(row[column]):
            return str(row[column])
    return None


def product_display_name(row: pd.Series) -> str:
    asin = product_key(row) or "unknown"
    for column in ("product_title", "business_name"):
        if column in row and not is_missing(row[column]):
            title = str(row[column]).strip()
            if title and title.lower() != asin.lower():
                return title
    return f"Unknown product (ASIN {asin[:12]})"


def normalize_review_keys(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if normalized.empty:
        normalized["_product_key"] = pd.Series(dtype="object")
        normalized["_month_key"] = pd.Series(dtype="object")
        return normalized

    def get_month(row: pd.Series) -> str | None:
        for column in ("review_month", "product_month", "business_month"):
            if column in row and not is_missing(row[column]):
                return month_key(row[column])
        return None

    normalized["_product_key"] = normalized.apply(product_key, axis=1)
    normalized["_month_key"] = normalized.apply(get_month, axis=1)
    return normalized


def filter_review_backed_cases(queue: pd.DataFrame, reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    queue_norm = normalize_review_keys(queue)
    reviews_norm = normalize_review_keys(reviews)
    if queue_norm.empty or reviews_norm.empty:
        return queue_norm.iloc[0:0].copy(), reviews_norm
    review_keys = reviews_norm[["_product_key", "_month_key"]].dropna().drop_duplicates()
    backed_queue = queue_norm.merge(review_keys, on=["_product_key", "_month_key"], how="inner")
    return backed_queue, reviews_norm


def signal_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row or is_missing(row[column]):
        return default
    try:
        return float(row[column])
    except Exception:
        return default


def anomaly_label(score: object) -> str:
    value = signal_value(pd.Series({"score": score}), "score")
    if value >= 0.90:
        return "Highly unusual"
    if value >= 0.70:
        return "Unusual"
    return "Lower priority"


def has_ingredient_list_pattern(text: str) -> bool:
    lower = text.lower()
    ingredient_terms = [
        "ingredient list",
        "ingredients:",
        "full ingredient",
        "butylene glycol",
        "glycerin",
        "hyaluronic acid",
        "ceramide",
        "disodium edta",
        "chlorphenesin",
    ]
    comma_count = text.count(",")
    return any(term in lower for term in ingredient_terms) or comma_count >= 18


def deterministic_explanation(row: pd.Series) -> str:
    parts: list[str] = []
    stars = signal_value(row, "stars")
    deviation = signal_value(row, "rating_deviation_from_product_average")
    word_count = signal_value(row, "word_count")
    reviewer_avg = signal_value(row, "reviewer_average_rating", -1)
    reviewer_count = signal_value(row, "reviewer_review_count")
    text = clean_review_text(row.get("review_text", row.get("text", "")))
    lower = text.lower()

    if stars >= 5 and word_count <= 12:
        parts.append("This is a very short 5-star review, which can resemble low-detail promotional activity.")
    elif stars <= 1 and word_count <= 12:
        parts.append("This is a very short 1-star review, which can indicate low-detail extreme feedback.")

    if deviation > 0.25:
        parts.append("The rating is higher than this product's usual rating.")
    elif deviation < -0.25:
        parts.append("The rating is lower than this product's usual rating.")

    if reviewer_count and reviewer_count <= 2:
        parts.append("This reviewer has limited review history in the observed data.")
    if reviewer_avg >= 4.5:
        parts.append("This reviewer usually gives high ratings.")
    elif 0 <= reviewer_avg <= 2.0:
        parts.append("This reviewer usually gives low ratings.")

    if signal_value(row, "generic_praise_flag") >= 1 or any(
        phrase in lower for phrase in ("great product", "love it", "works great", "highly recommend")
    ):
        parts.append("The wording includes generic praise that can appear in templated reviews.")
    if signal_value(row, "synthetic_templated_language_score") >= 0.50:
        parts.append("The language appears generic, templated, overly promotional, or low-specificity.")
    if signal_value(row, "repeated_character_flag") >= 1:
        parts.append("The text contains repeated characters that may indicate unnatural formatting.")
    if signal_value(row, "text_formatting_anomaly_flag") >= 1:
        parts.append("The text has formatting irregularities such as broken spacing or excessive line breaks.")
    if signal_value(row, "uppercase_share") > 0.45:
        parts.append("The review has unusually high capitalization.")
    if len(text) > 20 and " " not in text[:80]:
        parts.append("The text has little normal spacing, which can be a bot-like formatting signal.")
    if has_ingredient_list_pattern(text):
        parts.append("The review includes a long ingredient-list style block, which can be a copied product-detail or promotional-content signal rather than normal customer language.")

    if not parts:
        parts.append("This review was prioritized because multiple fraud-like signals are elevated for this product-month.")

    parts.append("This is not a final fraud verdict. It is a reason to review.")
    return " ".join(parts)


def safe_widget_key(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:180]


def load_saved_labels() -> dict[str, str]:
    if not LABEL_PATH.exists():
        return {}
    try:
        labels = pd.read_csv(LABEL_PATH)
        if labels.empty or "review_id" not in labels.columns or "label" not in labels.columns:
            return {}
        return (
            labels.dropna(subset=["review_id", "label"])
            .drop_duplicates("review_id", keep="last")
            .set_index("review_id")["label"]
            .to_dict()
        )
    except Exception:
        return {}


def save_label(review_id: str, label: str, notes: str, state_key: str) -> None:
    LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [{"saved_at": datetime.utcnow().isoformat(), "review_id": review_id, "label": label, "notes": notes}]
    )
    header = not LABEL_PATH.exists()
    row.to_csv(LABEL_PATH, mode="a", header=header, index=False)
    st.session_state[state_key] = True


def review_id_for(row: pd.Series, fallback: str) -> str:
    return str(row.get("review_id", fallback))


def metric_card(label: str, value: object, definition: str) -> None:
    st.metric(label, value)
    st.caption(definition)


def render_large_review_text(review_text: str) -> None:
    escaped = html.escape(review_text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            font-size: 1.2rem;
            line-height: 1.7;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(128, 128, 128, 0.08);
        ">
            {escaped}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_review_card(
    review: pd.Series,
    review_id: str,
    widget_key: str,
    saved_labels: dict[str, str],
    expanded: bool = True,
    title: str | None = None,
) -> None:
    stars = signal_value(review, "stars")
    score = signal_value(review, "anomaly_score")
    reviewed_label = saved_labels.get(review_id)

    header = title or f"{stars:g} star review - {anomaly_label(score)}"
    if reviewed_label:
        header = f"Reviewed: {reviewed_label} | {header}"

    with st.expander(header, expanded=expanded):
        st.markdown(f"### {product_display_name(review)}")
        st.caption(f"Product-month: {format_month(review.get('review_month', review.get('product_month')))}")
        st.caption(f"Rating: {stars:g}★")

        if reviewed_label:
            st.success(f"Reviewed: {reviewed_label}")

        if "synthetic_templated_language_score" in review.index:
            st.metric(
                "Synthetic / templated language signal",
                f"{signal_value(review, 'synthetic_templated_language_score'):.2f}",
            )
            st.caption(
                "This is not an AI detector; it is one weak prioritization signal for generic, templated, "
                "overly promotional, or low-specificity language."
            )

        st.markdown("*Why this review was flagged:*")
        st.write(deterministic_explanation(review))

        st.markdown("**Review text:**")
        review_text = clean_review_text(review.get("review_text", review.get("text", review.get("review_text_preview", ""))))
        render_large_review_text(review_text)

        with st.expander("Technical signals"):
            signal_columns = [column for column in TECHNICAL_SIGNAL_DEFINITIONS if column in review.index]
            st.dataframe(pd.DataFrame([{column: review[column] for column in signal_columns}]), use_container_width=True)

            st.markdown("**What these technical signals mean**")
            for column in signal_columns:
                st.caption(f"{column}: {TECHNICAL_SIGNAL_DEFINITIONS[column]}")

        st.caption("Labels are stored locally for review and threshold evaluation. They do not automatically retrain the model yet.")
        label = st.radio(
            "Moderator label",
            ["suspicious", "not suspicious", "unsure"],
            horizontal=True,
            key=f"label_{widget_key}",
        )
        notes = st.text_area("Notes", key=f"notes_{widget_key}")

        if st.button("Save label", key=f"save_{widget_key}"):
            save_label(review_id, label, notes, f"saved_{widget_key}")
            saved_labels[review_id] = label

        if st.session_state.get(f"saved_{widget_key}"):
            st.success(f"Review label saved locally. Reviewed: {label}")


st.title("Marketplace Integrity Monitor")
st.warning("**Please be patient: the dashboard may take 5–10 seconds to load the review queue.**")

st.markdown("## What to do")
st.markdown(
    """
1. Select a product-month case from the review queue.
2. Review the high-confidence review manipulation signals.
3. Label each review as suspicious, not suspicious, or unsure.
"""
)

st.write(
    "This tool identifies potentially suspicious review activity by combining anomaly detection, "
    "behavioral changes, and review-level signals. It helps moderators prioritize investigation. "
    "It does not make final decisions."
)
st.info(
    "This prototype uses the Amazon Reviews 2023 collection. The All_Beauty reviews include historical "
    "timestamps, which are replayed here to simulate reviews arriving over time."
)

with st.expander("How it works", expanded=True):
    st.markdown(
        "- Historical Amazon reviews are replayed as if new reviews arrive daily.\n"
        "- Reviews are scored for rating deviation, review length, reviewer behavior, and bot-like text signals.\n"
        "- Products are prioritized when high-confidence review-manipulation signals cluster in the same product-month.\n"
        "- Behavioral drift checks whether review volume or ratings changed sharply compared with prior periods.\n"
        "- Moderator labels are saved locally for evaluation, but do not automatically retrain the model."
    )

st.info(
    "Synthetic / templated language signal: This is not an AI detector. It flags language that appears "
    "generic, templated, overly promotional, or low-specificity. It is only one weak signal for review prioritization."
)

queue_raw = read_output_dir(QUEUE_PATH)
reviews_raw = read_output_dir(REVIEWS_PATH)
saved_labels = load_saved_labels()

if queue_raw.empty:
    st.info("No review queue found. Run the pipeline to generate moderation queue outputs.")
    st.stop()

queue, reviews = filter_review_backed_cases(queue_raw, reviews_raw)
if queue.empty:
    st.warning("No review-backed flagged cases found. Rerun the pipeline or check output keys.")
    st.stop()

mode = st.radio(
    "Review mode",
    ["Cases awaiting review", "Synthetic / templated language reviews"],
    horizontal=True,
)

if mode == "Cases awaiting review":
    sort_columns = [column for column in ["decision_score", "review_count"] if column in queue.columns]
    if sort_columns:
        queue = queue.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)

    labels = []
    for _, row in queue.iterrows():
        priority = signal_value(row, "decision_score")
        labels.append(f"{product_display_name(row)} | {format_month(row.get('review_month'))} | priority {priority:.2f}")

    st.subheader("Review queue")
    st.caption("Select a product-month case to review the strongest review-manipulation signals for that listing.")
    st.caption(f"Showing review-backed cases only. {len(queue):,} cases in queue.")

    selected_label = st.selectbox("Cases awaiting review", labels)
    selected = queue.iloc[labels.index(selected_label)]

    st.divider()
    st.subheader(product_display_name(selected))
    st.caption(f"Product-month: {format_month(selected.get('review_month'))}")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card(
            "Priority score",
            f"{signal_value(selected, 'decision_score'):.2f}",
            "Overall investigation priority for this product-month.",
        )
    with metric_cols[1]:
        metric_card(
            "Supporting anomaly score",
            f"{signal_value(selected, 'max_anomaly_score'):.2f}",
            "Supporting statistical unusualness. Review inclusion is driven by fraud-oriented signals.",
        )
    with metric_cols[2]:
        metric_card(
            "Share of unusual reviews",
            f"{signal_value(selected, 'suspicious_review_density'):.2f}",
            "Share of reviews in this product-month that crossed the anomaly threshold.",
        )
    with metric_cols[3]:
        metric_card(
            "Change in review patterns",
            f"{signal_value(selected, 'behavioral_drift_score'):.2f}",
            "Change from prior month in review volume, average rating, or extreme-rating share.",
        )

    filtered_reviews = reviews[
        (reviews["_product_key"] == selected["_product_key"])
        & (reviews["_month_key"] == selected["_month_key"])
    ].copy()

    sort_cols = [column for column in ["fraud_like_score", "fraud_signal_count", "anomaly_score"] if column in filtered_reviews.columns]
    if sort_cols:
        filtered_reviews = filtered_reviews.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    st.caption(f"{len(filtered_reviews):,} high-confidence review signals for selected case.")
    st.subheader("High-confidence review manipulation signals")

    case_key = safe_widget_key(f"{selected['_product_key']}_{selected['_month_key']}")
    for display_number, (row_idx, review) in enumerate(filtered_reviews.iterrows(), start=1):
        review_id = review_id_for(review, f"{selected['_product_key']}_{selected['_month_key']}_{display_number}")
        widget_key = safe_widget_key(f"case_{case_key}_{row_idx}_{display_number}_{review_id}")
        render_review_card(
            review=review,
            review_id=review_id,
            widget_key=widget_key,
            saved_labels=saved_labels,
            title=f"Review #{display_number}: {signal_value(review, 'stars'):g} star review - {anomaly_label(signal_value(review, 'anomaly_score'))}",
        )

else:
    st.subheader("Synthetic / templated language reviews")
    st.caption(
        "Use this to explore reviews that look templated or generic across products. "
        "This is not an AI detector."
    )

    if "synthetic_templated_language_score" not in reviews.columns:
        st.info("No synthetic / templated language score found in the current review output.")
    else:
        synthetic_reviews = (
            reviews.dropna(subset=["synthetic_templated_language_score"])
            .sort_values("synthetic_templated_language_score", ascending=False)
            .head(200)
        )

        synthetic_labels = []
        synthetic_lookup: dict[str, int] = {}

        for row_idx, row in synthetic_reviews.iterrows():
            score = signal_value(row, "synthetic_templated_language_score")
            stars = signal_value(row, "stars")
            preview = clean_review_text(row.get("review_text", row.get("text", row.get("review_text_preview", ""))))[:90]
            name = product_display_name(row)
            rid = review_id_for(row, f"synthetic_{row_idx}")
            reviewed = saved_labels.get(rid)
            reviewed_prefix = f"Reviewed: {reviewed} | " if reviewed else ""
            label = f"{reviewed_prefix}{score:.2f} | {stars:g}★ | {name} | {preview}"
            synthetic_labels.append(label)
            synthetic_lookup[label] = row_idx

        selected_synthetic_label = st.selectbox(
            "Browse reviews with highest synthetic / templated language signal",
            synthetic_labels,
        )

        selected_idx = synthetic_lookup[selected_synthetic_label]
        synth = synthetic_reviews.loc[selected_idx]
        synth_review_id = review_id_for(synth, f"synthetic_{selected_idx}")
        synth_key = safe_widget_key(f"synthetic_{selected_idx}_{synth_review_id}")

        render_review_card(
            review=synth,
            review_id=synth_review_id,
            widget_key=synth_key,
            saved_labels=saved_labels,
            title=f"Synthetic / templated signal {signal_value(synth, 'synthetic_templated_language_score'):.2f} | {signal_value(synth, 'stars'):g}★",
        )

st.divider()

with st.expander("Methodology", expanded=False):
    st.markdown(
        """
**Summary**  
This system prioritizes likely review manipulation by combining multiple weak signals and surfacing only high-confidence cases for human review.

This prototype is a local Trust & Safety moderation workflow built on the Amazon Reviews 2023 All_Beauty data.

**Data and replay design**
- The app uses historical Amazon review timestamps and replays them as if reviews are arriving over time.
- The current category is All_Beauty, chosen because it is small enough to run locally but still contains realistic marketplace review patterns.
- Product-months are used as the main investigation unit, so reviewers can inspect clusters of suspicious activity rather than isolated records.

**Scoring approach**
- The system combines product-level behavior, reviewer behavior, text features, and anomaly detection.
- The Isolation Forest model provides a supporting anomaly score, but anomaly score alone does not drive the reviewer queue.
- The reviewer-facing queue is tuned for precision over recall, meaning it shows fewer cases that are more likely to be worth a moderator’s time.

**Fraud-like signals**
Reviews are prioritized when multiple manipulation-style signals appear together, including:
- very short 5-star praise
- generic or templated wording
- repeated or unusual formatting
- copied product-detail style text, such as unusually long ingredient-list blocks
- suspicious reviewer rating patterns
- product-month bursts or behavioral drift
- synthetic / templated language signals

**Synthetic / templated language**
The synthetic / templated language score is not an AI detector. It is a weak prioritization signal for generic, overly promotional, low-specificity, or template-like review text.

**Human-in-the-loop review**
Moderator labels are saved locally for review, false-positive analysis, and threshold tuning. In this prototype, labels do not automatically retrain the model.

**System design**
The project is intentionally local and portfolio-sized: PySpark for feature engineering, pandas CSV outputs for Windows compatibility, and Streamlit for the moderation interface.
        """
    )

st.markdown(
    "Built by [David Rauch](https://github.com/davidwrauch) as a prototype marketplace integrity system."
)