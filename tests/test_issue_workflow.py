"""The scheduled job that issues the daily forecasts (.github/workflows/issue.yml). The workflow is read as
text, so this needs no YAML parser."""

import re
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "issue.yml"


def test_job_issues_into_the_hapi_checkout():
    text = WORKFLOW.read_text()
    assert re.search(r"cron: '\d+ \d+ \* \* \*'", text), "runs once a day"
    assert 'python scripts/issue.py --model "$model" --out "$HAPI_DATA"' in text
    env = dict(re.findall(r"^\s+(CAUDAL_DATA|HAPI_DATA): (.+)$", text, re.M))
    assert env["CAUDAL_DATA"] == env["HAPI_DATA"] and env["HAPI_DATA"].endswith("/hapi-data/data")
    assert set(re.findall(r"token: (.+)", text)) == {"${{ secrets.HAPI_TOKEN }}"}, "the push right comes only from the secret"
    assert "git add data/forecasts data/forcing" in text, "the forcing that issued the forecasts is committed with them"


def test_job_issues_the_current_daily_model():
    text = WORKFLOW.read_text()
    loop = re.search(r"for model in (.+); do", text)
    assert loop and loop.group(1).split() == ["quantile_inflow_v2_s0"], "v2 replaced v1 on 2026-11-19"
    assert "|| failed=1" in text and "exit $failed" in text, "a model that fails leaves the job red"
