#!/usr/bin/env python3
"""Greptile review-budget counter — the CodeRabbit fallback's monthly cap (CLAUDE.md §5.1).

Greptile is this repo's *fallback* reviewer: CodeRabbit is the primary review gate, and
Greptile is invoked only when CodeRabbit is rate- or quota-limited. Two properties are
enforced outside this script and one is enforced by it:

* **Never fires automatically** — ``.greptile/config.json`` pins ``skipReview:"AUTOMATIC"``,
  which disables automatic PR review while leaving the manual ``@greptileai`` comment
  trigger working. (Greptile config, not this script.)
* **Capped at 16 reviews per month** — Greptile exposes **no native per-repo review quota**
  (its config has strictness/label/branch/author filters and ``fileChangeLimit``, but no
  numeric monthly cap), so the cap has to be enforced repo-side. That is this script.

Counting unit — why the *trigger*, not Greptile's output
--------------------------------------------------------
Greptile's output surface is **not** a stable counting unit; sampling real public PRs shows
all three of these shapes:

* a sticky ``<!-- greptile-status -->`` issue comment that is **edited in place** across
  re-reviews (observed: 2 reviews, 1 comment) — counting comments *undercounts*;
* a ``### Greptile Summary`` issue comment with **zero** ``/pulls/N/reviews`` objects;
* N review objects + N inline comments that scale with the number of findings — counting
  those *overcounts*.

Because automatic review is disabled, every review is preceded by exactly one human
``@greptileai`` comment, so **one review == one trigger comment**. That is immutable,
timestamped, and independent of whichever output shape Greptile happens to use.

The count is **re-derived from the GitHub API on every call** — never from a ledger file
this repo writes about itself, which would drift the moment an invocation went unlogged
(CLAUDE.md §10.3: never emit a number that isn't measured).

Auto-fire alarm (the gate is also the detector)
-----------------------------------------------
Greptile bot activity on a PR with **no in-period trigger** means a review happened that
nobody asked for — i.e. ``skipReview`` is not in force (dashboard/org override, config not
on the default branch, malformed JSON). Those are counted as consumed budget *and* raised
as a loud warning, so the budget gate doubles as the breach detector for the config.

Known limitation
----------------
Auto-fire matching is per-thread, not per-event: if Greptile auto-fires on a PR and that
same PR is *later* triggered manually in the same period, the bot's activity is attributed
to the trigger and the pair counts as **one**, not two. Charging both would double-count
the ordinary trigger-then-response case, which is far more common. The residual undercount
is bounded by the number of auto-fires — and an auto-fire on an untriggered thread is still
caught, charged, and reported, so the condition never goes unnoticed.

Fail-closed
-----------
Every non-200, transport failure, or truncated pagination raises ``BudgetError`` and exits
**3** — the script never prints a remaining-budget number it did not measure. "I could not
measure" must never be read as "0 used, 16 remaining" (CLAUDE.md §10.3). Bot authors are
matched by the ``greptile`` login prefix, not one hardcoded login, because Greptile owns
more than one bot identity (``greptile-apps[bot]``, id 165735046, the currently active
reviewer; and ``greptile[bot]``, id 271099122, registered at github.com/apps/greptile) —
pinning a single login would silently return 0 forever if the other one posts.

Exit codes:
    0  budget available    (prints remaining)
    2  budget exhausted    (used >= limit) — do NOT invoke Greptile
    3  could not measure   (fail-closed) — do NOT invoke Greptile
    4  usage error         (bad CLI args; nothing was measured)

Only 0 means "invoke". 4 exists because argparse's own usage-error exit is 2, which would
otherwise be read as "this month's budget is spent" on nothing worse than a mistyped flag.

Usage:
    python scripts/greptile_budget.py                  # human summary + JSON
    python scripts/greptile_budget.py --json out.json
    GH_TOKEN=$(gh auth token --user bioedca) python scripts/greptile_budget.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

# --- Pinned budget policy -------------------------------------------------------------
# The cap and its anchor date. The budget period is monthly, half-open, anchored on the
# anchor's day-of-month: period k == [anchor+k months, anchor+(k+1) months).
DEFAULT_REPO = "bioedca/tbox-finder"
DEFAULT_LIMIT = 16
DEFAULT_ANCHOR = "2026-08-03"  # the day the Greptile fallback was configured

# Greptile's manual triggers. BOTH handles must be matched: `@greptileai` is the documented
# one, and `@greptile-apps` is the one Greptile itself offers to bypass `fileChangeLimit`
# ("Bypass the limit by tagging `@greptile-apps` to review." — observed verbatim on PR #98).
# Counting only the documented handle would miss a real invocation, and an uncounted review
# is permissive in exactly the direction that overruns the cap.
# Matched case-insensitively and required to be followed by a non-word character (or end of
# string) so `@greptileaifoo` does not count.
_TRIGGER_RE = re.compile(r"@greptile(?:ai|-apps)(?!\w)", re.IGNORECASE)

# Greptile owns >1 bot identity; match the family by prefix rather than pinning one login.
_BOT_LOGIN_PREFIX = "greptile"

USER_AGENT = "tbox-finder-greptile-budget/1.0 (+CLAUDE.md-5.1)"
USAGE_ERROR_EXIT = 4  # kept outside the 0/2/3 budget contract (argparse itself exits 2)
_MAX_PAGES = 100  # 100 pages x 100 comments; exceeding this raises rather than truncating


class BudgetError(RuntimeError):
    """A measurement failure that must NOT be read as 'no reviews used'.

    A bad token, an outage, a rate-limit, or a truncated page would otherwise report
    ``used=0`` — indistinguishable from a genuinely unused budget, and permissive in
    exactly the direction that lets the cap be blown through (CLAUDE.md §10.3).
    """


# --- Period arithmetic (pure) ---------------------------------------------------------


def _add_months(dt: datetime, months: int) -> datetime:
    """Advance `dt` by whole months, keeping the day-of-month.

    The anchor day is 3, which exists in every month, so no clamping is required; a day
    that could overflow (29-31) is rejected by `parse_anchor` rather than silently clamped.
    """
    total = dt.month - 1 + months
    year = dt.year + total // 12
    month = total % 12 + 1
    return dt.replace(year=year, month=month)


def parse_anchor(anchor: str) -> datetime:
    """Parse the YYYY-MM-DD budget anchor as midnight UTC.

    Rejects days 29-31: those do not exist in every month, so monthly stepping would need
    a clamping rule, and a clamped boundary silently changes period lengths.
    """
    try:
        dt = datetime.strptime(anchor, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        # Escaping as a bare ValueError would exit 1 — outside the 0/2/3 contract, so a
        # caller keying on those codes could not tell it apart from a crash.
        raise BudgetError(f"unparseable anchor {anchor!r} (want YYYY-MM-DD): {exc}") from exc
    if dt.day > 28:
        raise BudgetError(
            f"anchor day {dt.day} is not present in every month; "
            "use a day in 1-28 so monthly periods need no clamping"
        )
    return dt


def current_period(now: datetime, anchor: datetime) -> tuple[datetime, datetime]:
    """Return the half-open [start, end) UTC period containing `now`.

    All arithmetic is UTC because GitHub timestamps are UTC (`Z`); using a local clock
    would move reviews across a boundary depending on where the script happens to run.
    """
    if now < anchor:
        raise BudgetError(f"now ({now.isoformat()}) precedes the budget anchor {anchor.date()}")
    # Step whole months from the anchor. The loop runs once per elapsed month; cheap, and
    # exact (no 30.44-day drift that a division-based estimate would accumulate).
    k = (now.year - anchor.year) * 12 + (now.month - anchor.month)
    if _add_months(anchor, k) > now:
        k -= 1
    return _add_months(anchor, k), _add_months(anchor, k + 1)


# --- HTTP ------------------------------------------------------------------------------


def _get(url: str, token: str | None) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers or {})
    except urllib.error.URLError as exc:
        return 0, str(exc.reason).encode(), {}
    except OSError as exc:
        # A read timeout mid-`resp.read()` surfaces as a bare TimeoutError, which urllib
        # does NOT wrap in URLError. Uncaught it would escape the BudgetError path and exit
        # 1, outside the 0/2/3 contract. Ordering matters: HTTPError < URLError < OSError,
        # so this must stay last or it would swallow the two more specific cases.
        return 0, f"{type(exc).__name__}: {exc}".encode(), {}


def _next_link(headers: dict[str, str]) -> str | None:
    """Extract the rel="next" URL from a GitHub Link header, if any."""
    for part in (headers.get("Link") or headers.get("link") or "").split(","):
        seg = part.split(";")
        if len(seg) >= 2 and 'rel="next"' in seg[1] and seg[0].strip().startswith("<"):
            return seg[0].strip()[1:-1]
    return None


def fetch_issue_comments(repo: str, since: datetime, token: str | None) -> list[dict]:
    """All repo issue/PR comments updated at or after `since`, following pagination.

    Uses the repository-wide comments endpoint (one paginated REST walk) rather than the
    Search API, which is eventually consistent and capped at 1000 results — a search-index
    lag would under-report a review triggered minutes ago and hand back budget that is
    already spent. `since` filters on *updated_at*, a superset of the comments *created*
    in-period (created >= since implies updated >= since); the created_at window is applied
    exactly in `summarise`.
    """
    url = (
        f"https://api.github.com/repos/{repo}/issues/comments"
        f"?per_page=100&since={urllib.parse.quote(since.strftime('%Y-%m-%dT%H:%M:%SZ'))}"
    )
    out: list[dict] = []
    for _ in range(_MAX_PAGES):
        code, body, headers = _get(url, token)
        if code != 200:
            raise BudgetError(f"GET {url} -> HTTP {code}: {body[:200]!r}")
        try:
            batch = json.loads(body)
        except json.JSONDecodeError as exc:
            raise BudgetError(f"GET {url} -> unparseable JSON: {exc}") from exc
        if not isinstance(batch, list):
            raise BudgetError(f"GET {url} -> expected a JSON array, got {type(batch).__name__}")
        out.extend(batch)
        nxt = _next_link(headers)
        if not nxt:
            return out
        url = nxt
    raise BudgetError(
        f"pagination exceeded {_MAX_PAGES} pages for {repo}; refusing to report a "
        "count derived from a truncated comment list"
    )


# --- Pure core -------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime:
    """Parse a GitHub UTC timestamp, failing closed on anything unexpected.

    Raises `BudgetError`, not `ValueError`: a comment whose `created_at` GitHub renders in
    an unexpected shape must route through the fail-closed path (exit 3) rather than
    crashing out of the 0/2/3 exit contract, and must never be silently skipped — a
    skipped comment is an uncounted review.
    """
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError) as exc:
        raise BudgetError(f"unparseable UTC timestamp {value!r}: {exc}") from exc


def _is_greptile_bot(user: dict) -> bool:
    return (user.get("type") == "Bot") and str(user.get("login", "")).lower().startswith(
        _BOT_LOGIN_PREFIX
    )


def summarise(
    comments: list[dict],
    *,
    period_start: datetime,
    period_end: datetime,
    limit: int,
) -> dict:
    """Network-free core: turn raw issue comments into the budget verdict.

    `used` = manual ``@greptileai`` triggers in-period, plus any thread where the Greptile
    bot posted in-period with **no** in-period trigger. That second term is an auto-fired
    review: it consumes budget (conservative) and raises `auto_fire_suspected`, because
    with ``skipReview:"AUTOMATIC"`` in force it should be structurally impossible.
    """
    triggers: list[dict] = []
    bot_threads: dict[str, str] = {}  # issue_url -> first in-period bot comment html_url
    observed_bot_logins: set[str] = set()

    for c in comments:
        user = c.get("user") or {}
        created_raw = c.get("created_at")
        if not created_raw:
            raise BudgetError(f"comment {c.get('id')!r} has no created_at; cannot place it")
        created = _parse_ts(created_raw)
        created_in_period = period_start <= created < period_end
        thread = c.get("issue_url") or ""
        if _is_greptile_bot(user):
            # Greptile EDITS one sticky <!-- greptile-status --> comment in place across
            # re-reviews rather than posting a new one, so a run that happened in THIS
            # period can carry a created_at from a PREVIOUS one. Placing bot comments by
            # created_at alone would make an auto-fired re-review on an older PR invisible
            # — an undercount, permissive in the direction that overruns the cap. `since=`
            # filters on updated_at, so such a comment is already in the fetched page.
            updated = _parse_ts(c.get("updated_at") or created_raw)
            if not (created_in_period or period_start <= updated < period_end):
                continue
            observed_bot_logins.add(user.get("login", ""))
            bot_threads.setdefault(thread, c.get("html_url", ""))
        elif not created_in_period:
            # A trigger is placed by creation only: editing an old comment does not
            # re-trigger a review, so its edit must not spend budget.
            continue
        elif user.get("type") != "Bot" and _TRIGGER_RE.search(c.get("body") or ""):
            triggers.append(
                {
                    "thread": thread,
                    "author": user.get("login", ""),
                    "created_at": created_raw,
                    "url": c.get("html_url", ""),
                }
            )

    triggered_threads = {t["thread"] for t in triggers}
    unmatched = sorted(url for th, url in bot_threads.items() if th not in triggered_threads)
    used = len(triggers) + len(unmatched)
    return {
        "period_start": period_start.isoformat().replace("+00:00", "Z"),
        "period_end": period_end.isoformat().replace("+00:00", "Z"),
        "limit": limit,
        "used": used,
        "remaining": max(0, limit - used),
        "exhausted": used >= limit,
        "triggers": triggers,
        "trigger_count": len(triggers),
        "auto_fire_suspected": unmatched,
        "auto_fire_count": len(unmatched),
        "observed_bot_logins": sorted(observed_bot_logins),
    }


def render(report: dict) -> str:
    lines = [
        "",
        "Greptile review budget — tbox-finder (CodeRabbit fallback, CLAUDE.md §5.1)",
        "=" * 74,
        f"  period      : {report['period_start']}  ->  {report['period_end']}",
        f"  used        : {report['used']} / {report['limit']}"
        f"   ({report['trigger_count']} manual trigger(s)"
        f" + {report['auto_fire_count']} unrequested)",
        f"  remaining   : {report['remaining']}",
        f"  verdict     : {'EXHAUSTED — do not invoke' if report['exhausted'] else 'available'}",
    ]
    for t in report["triggers"]:
        lines.append(f"    - {t['created_at']}  @{t['author']}  {t['url']}")
    if report["auto_fire_count"]:
        lines += [
            "",
            f"  !! {report['auto_fire_count']} Greptile review(s) with NO manual trigger.",
            '     skipReview:"AUTOMATIC" is not in force — check the Greptile dashboard /',
            "     org-enforced rules and that .greptile/config.json is on the default branch.",
        ]
        lines += [f"       {u}" for u in report["auto_fire_suspected"]]
    lines.append("")
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    """An argparse type for `--limit`, so a nonsensical cap is a usage error.

    Without this, `--limit 0` makes `used >= limit` true on an empty period and the script
    reports exit 2 — "this month's budget is spent" — when nothing was spent and the
    argument was simply wrong. Same conflation the argparse-exit-2 remap fixes.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--limit must be >= 1, got {parsed}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Greptile monthly review-budget counter.")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default {DEFAULT_REPO})")
    ap.add_argument(
        "--limit", type=_positive_int, default=DEFAULT_LIMIT, help="reviews per period (>0)"
    )
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR, help="YYYY-MM-DD period anchor")
    ap.add_argument("--now", help="override 'now' as YYYY-MM-DDTHH:MM:SSZ (testing)")
    ap.add_argument("--json", metavar="PATH", help="also write the JSON report to PATH")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on a usage error, which collides with this script's "exhausted".
        # A mistyped flag would otherwise read as "the month's budget is spent". Remap to a
        # code outside the 0/2/3 budget contract; --help/--version (code 0) stay 0.
        if exc.code in (0, None):
            return 0
        print("greptile-budget: usage error; no budget was measured.", file=sys.stderr)
        return USAGE_ERROR_EXIT

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    try:
        anchor = parse_anchor(args.anchor)
        now = _parse_ts(args.now) if args.now else datetime.now(UTC)
        start, end = current_period(now, anchor)
        comments = fetch_issue_comments(args.repo, start, token)
        report = summarise(comments, period_start=start, period_end=end, limit=args.limit)
    except BudgetError as exc:
        # Fail closed: no remaining-budget number is printed, and the exit code is distinct
        # from "exhausted" so a caller cannot conflate "unmeasured" with "measured zero".
        print(f"greptile-budget: COULD NOT MEASURE — {exc}", file=sys.stderr)
        print(
            "greptile-budget: refusing to report a budget; do NOT invoke Greptile.", file=sys.stderr
        )
        return 3

    report["repo"] = args.repo
    report["authenticated"] = bool(token)
    print(render(report))
    payload = json.dumps(report, indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(payload + "\n")
        print(f"[wrote JSON report → {args.json}]")
    return 2 if report["exhausted"] else 0


if __name__ == "__main__":
    sys.exit(main())
