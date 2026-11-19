"""The scheduled job that issues the hourly flow forecasts (.github/workflows/issue_hourly.yml). The workflow
is read as text, so this needs no YAML parser."""

import re
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "issue_hourly.yml"


def test_job_issues_every_hour_into_the_hapi_checkout():
    text = WORKFLOW.read_text()
    assert re.search(r"cron: '\d+ \* \* \* \*'", text), "every hour, so each observed hour has a forecast made the hour before"
    loop = re.search(r"for model in (.+); do", text)
    assert loop and loop.group(1).split() == ["quantile_flow_hourly_v1_s0", "quantile_flow_hourly_v2_s0"]
    assert 'python scripts/issue_hourly.py --model "$model" --out "$HAPI_DATA" || failed=1' in text
    env = dict(re.findall(r"^\s+(CAUDAL_DATA|HAPI_DATA): (.+)$", text, re.M))
    assert env["CAUDAL_DATA"] == env["HAPI_DATA"] and env["HAPI_DATA"].endswith("/hapi-data/data")
    assert set(re.findall(r"token: (.+)", text)) == {"${{ secrets.HAPI_TOKEN }}"}, "the push right comes only from the secret"
    assert re.findall(r"git add (.+)", text) == ["data/forecasts"], "nothing else on the branch is touched"
