# Evaluation Report: gemini / gemini-3.8-flash

> [!WARNING]
> **INVALID RUN**: Suite aborted due to 3 consecutive rate limits with no backup provider available.

- **Timestamp**: 2026-10-02 08:36:15 UTC
- **Provider**: `gemini`
- **Model**: `gemini-3.8-flash`
- **Cost**: `cost: estimated/unverified`
- **Git Commit**: `e282dcc`
- **Overall Pass Rate**: **0.0%** (0/0)
- **Rate Limited Runs (excluded from denominator)**: **3**
- **Provider Unavailable Runs (excluded from denominator)**: **0**
- **Infrastructure Errors**: **0**
- **Safety Failures**: **0**

## Summary Metrics

| Metric | Value |
| :--- | :--- |
| Average Steps | 0.0 |
| Average Cost (USD) | cost: estimated/unverified |
| Average Wall Time | 13.67s |

### Performance by Provider

| Provider | Runs | Rate Limited | Provider Unavailable | Infra Errors | Passed / Eval Total | Pass Rate | Avg Cost ($) | Avg Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemini` | 3 | 3 | 0 | 0 | 0/0 | 0.0% | $0.0000 | 13.67s |

### Performance by Model Used

| Model Used | Runs | Provider Unavailable | Infra Errors | Passed / Eval Total | Success Rate | Avg Cost ($) | Avg Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-flash` | 3 | 0 | 0 | 0/0 | 0.0% | cost: estimated/unverified | 13.67s |

### Pass Rate by Category

| Category | Passed / Total | Pass Rate |
| :--- | :--- | :--- |
| Functional | 0/2 | 0.0% |
| Language | 0/1 | 0.0% |

## Detailed Results

| Test ID | Category | Result | Actual Status | Steps | Cost ($) | Time (s) | Failure Category | Detail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `login_en` | functional | **FAIL** | `error` | 0 | $0.0000 | 14.17s | rate_limited | Expected URL to contain '/inventory.html', but was 'https://www.saucedemo.com/' |
| `login_hi` | language | **FAIL** | `error` | 0 | $0.0000 | 13.57s | rate_limited | Expected URL to contain '/inventory.html', but was 'https://www.saucedemo.com/' |
| `cheapest_item` | functional | **FAIL** | `error` | 0 | $0.0000 | 13.26s | rate_limited | No items found on inventory page. |
