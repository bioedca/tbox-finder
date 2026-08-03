"""Unit tests for the Greptile monthly review-budget counter (CLAUDE.md §5.1, §10.3).

Exercises the network-free core of ``scripts/greptile_budget.py`` — the UTC period
arithmetic, the trigger/bot classification, and the fail-closed guards — against fixture
comment payloads shaped like real GitHub ``/repos/{o}/{r}/issues/comments`` items.

Two properties this suite is built to catch, both learned from real Greptile behaviour:

* **The sticky-comment undercount.** Greptile edits one ``<!-- greptile-status -->``
  comment in place across re-reviews (observed on a real PR: 2 reviews, 1 comment), so a
  counter keyed on Greptile's *output* reports 1 for 2 invocations. ``test_two_triggers_
  one_thread_*`` locks the trigger-keyed count that survives this.
* **Permissive inversion.** A counter that flips its exhausted/available verdict is far
  more dangerous than one that returns 0. Fixtures are deliberately **asymmetric** (3
  triggers vs 1 auto-fire vs limit 5) and assert *identity* — which threads were counted —
  not just totals, so swapping the two branches cannot keep the suite green.

Every ``pytest.raises`` here is paired with a positive control asserting the neighbouring
clean input still succeeds, so a guard that raised on *everything* would fail the suite.
Stdlib + pytest only.
"""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "greptile_budget.py"

_spec = importlib.util.spec_from_file_location("greptile_budget", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
# Register before exec so dataclasses/annotations can resolve the module (py3.12/3.13).
sys.modules["greptile_budget"] = mod
_spec.loader.exec_module(mod)

ANCHOR = mod.parse_anchor("2026-08-03")
P_START, P_END = mod.current_period(mod._parse_ts("2026-08-10T00:00:00Z"), ANCHOR)

THREAD_A = "https://api.github.com/repos/bioedca/tbox-finder/issues/101"
THREAD_B = "https://api.github.com/repos/bioedca/tbox-finder/issues/102"
THREAD_C = "https://api.github.com/repos/bioedca/tbox-finder/issues/103"


def _comment(*, thread, login, kind, body="", at="2026-08-10T12:00:00Z", cid=1):
    """One /issues/comments item. `kind` is 'User' or 'Bot' (the GitHub user.type)."""
    return {
        "id": cid,
        "issue_url": thread,
        "created_at": at,
        "html_url": f"{thread}#issuecomment-{cid}",
        "user": {"login": login, "type": kind},
        "body": body,
    }


def _trigger(thread, *, at="2026-08-10T12:00:00Z", cid=1, login="bioedca"):
    return _comment(thread=thread, login=login, kind="User", body="@greptileai", at=at, cid=cid)


def _bot(thread, *, at="2026-08-10T12:05:00Z", cid=2, login="greptile-apps[bot]"):
    return _comment(
        thread=thread, login=login, kind="Bot", body="### Greptile Summary", at=at, cid=cid
    )


def _summarise(comments, limit=5):
    return mod.summarise(comments, period_start=P_START, period_end=P_END, limit=limit)


def test_script_present():
    assert _SCRIPT.is_file()


# --- Period arithmetic -----------------------------------------------------------------


def test_period_anchored_on_the_configured_day():
    start, end = mod.current_period(mod._parse_ts("2026-08-03T00:00:00Z"), ANCHOR)
    assert start == datetime(2026, 8, 3, tzinfo=UTC)
    assert end == datetime(2026, 9, 3, tzinfo=UTC)


@pytest.mark.parametrize(
    ("now", "expect_start", "expect_end"),
    [
        # Half-open: the instant before the next anchor is still the first period ...
        ("2026-09-02T23:59:59Z", (2026, 8, 3), (2026, 9, 3)),
        # ... and the anchor instant itself begins the next one.
        ("2026-09-03T00:00:00Z", (2026, 9, 3), (2026, 10, 3)),
        ("2027-01-15T08:00:00Z", (2027, 1, 3), (2027, 2, 3)),
        # Year rollover must not reset to month 13 or drift by the 30.44-day average.
        ("2026-12-20T00:00:00Z", (2026, 12, 3), (2027, 1, 3)),
    ],
)
def test_period_boundaries_are_half_open_and_roll_over(now, expect_start, expect_end):
    start, end = mod.current_period(mod._parse_ts(now), ANCHOR)
    assert (start.year, start.month, start.day) == expect_start
    assert (end.year, end.month, end.day) == expect_end
    assert start < end


def test_now_before_anchor_refuses_but_on_anchor_succeeds():
    with pytest.raises(mod.BudgetError):
        mod.current_period(mod._parse_ts("2026-08-02T23:59:59Z"), ANCHOR)
    # Positive control: one second later is fine, so the guard is not refusing everything.
    assert mod.current_period(mod._parse_ts("2026-08-03T00:00:00Z"), ANCHOR)[0] == ANCHOR


def test_anchor_day_that_is_not_in_every_month_is_refused():
    for bad in ("2026-08-29", "2026-08-31"):
        with pytest.raises(mod.BudgetError):
            mod.parse_anchor(bad)
    # Positive control: day 28 exists in every month and must be accepted.
    assert mod.parse_anchor("2026-08-28").day == 28


# --- Counting unit ---------------------------------------------------------------------


def test_counts_one_trigger_per_invocation_and_names_them():
    report = _summarise([_trigger(THREAD_A, cid=1), _trigger(THREAD_B, cid=2)])
    assert report["used"] == 2
    assert report["remaining"] == 3
    assert report["exhausted"] is False
    # Identity, not just the total: a counter that tallied the wrong comments would pass a
    # bare `== 2` (see MEMORY: symmetric-count fixtures are blind to inversions).
    assert {t["thread"] for t in report["triggers"]} == {THREAD_A, THREAD_B}


def test_two_triggers_one_thread_count_twice_despite_a_single_sticky_bot_comment():
    """The real-world regression: Greptile edits one status comment across re-reviews.

    Two invocations on the same PR produce two trigger comments but only ONE Greptile
    comment. Counting Greptile's output would report 1 — a 2x undercount, permissive in
    exactly the direction that overruns the cap.
    """
    report = _summarise(
        [
            _trigger(THREAD_A, at="2026-08-10T12:00:00Z", cid=1),
            _trigger(THREAD_A, at="2026-08-14T09:00:00Z", cid=3),
            _bot(THREAD_A, cid=2),  # the single, in-place-edited sticky comment
        ]
    )
    assert report["used"] == 2
    assert report["auto_fire_count"] == 0


def test_comments_outside_the_period_are_excluded_on_both_sides():
    report = _summarise(
        [
            _trigger(THREAD_A, at="2026-08-02T23:59:59Z", cid=1),  # before start
            _trigger(THREAD_B, at="2026-09-03T00:00:00Z", cid=2),  # on next start
            _trigger(THREAD_C, at="2026-08-15T00:00:00Z", cid=3),  # inside
        ]
    )
    assert report["used"] == 1
    assert [t["thread"] for t in report["triggers"]] == [THREAD_C]


def test_a_bot_quoting_the_trigger_does_not_consume_budget():
    """Only humans trigger. A bot echoing '@greptileai' must not be counted as a trigger."""
    report = _summarise(
        [_comment(thread=THREAD_A, login="coderabbitai[bot]", kind="Bot", body="@greptileai")]
    )
    assert report["trigger_count"] == 0
    # ... and a non-Greptile bot is not mistaken for Greptile activity either.
    assert report["auto_fire_count"] == 0
    assert report["used"] == 0


@pytest.mark.parametrize(
    "body",
    [
        "@greptileai",
        "@greptileai review",
        "@greptile-apps",  # the handle Greptile offers to bypass fileChangeLimit
        "@greptile-apps review with strictness 1",
        "please take a look @GreptileAI",  # case-insensitive, mid-sentence
    ],
)
def test_both_trigger_handles_consume_budget(body):
    """Counting only @greptileai would miss a real invocation made via @greptile-apps."""
    report = _summarise([_comment(thread=THREAD_A, login="bioedca", kind="User", body=body)])
    assert report["used"] == 1, body


@pytest.mark.parametrize("body", ["@greptileaifoo", "@greptile-appsfoo", "@greptile", "greptileai"])
def test_near_miss_mentions_do_not_consume_budget(body):
    assert (
        _summarise([_comment(thread=THREAD_A, login="bioedca", kind="User", body=body)])["used"]
        == 0
    ), body


def test_trigger_must_be_the_whole_mention():
    assert (
        _summarise(
            [_comment(thread=THREAD_A, login="bioedca", kind="User", body="@greptileaifoo")]
        )["used"]
        == 0
    )
    # Positive control: the bare mention, and the mention with an argument, both count.
    assert _summarise([_trigger(THREAD_A)])["used"] == 1
    assert (
        _summarise(
            [_comment(thread=THREAD_A, login="bioedca", kind="User", body="@greptileai review")]
        )["used"]
        == 1
    )


# --- Auto-fire alarm (the gate is also the config-breach detector) ----------------------


def test_greptile_activity_without_a_trigger_is_charged_and_flagged():
    report = _summarise([_bot(THREAD_B, cid=9)])
    assert report["auto_fire_count"] == 1
    assert report["auto_fire_suspected"] == [f"{THREAD_B}#issuecomment-9"]
    assert report["used"] == 1, "an unrequested review still consumes the monthly cap"


def test_triggered_thread_is_not_flagged_as_auto_fire():
    """Positive control for the alarm: the same bot comment, now with its trigger."""
    report = _summarise([_trigger(THREAD_B, cid=8), _bot(THREAD_B, cid=9)])
    assert report["auto_fire_count"] == 0
    assert report["used"] == 1, "trigger + its own response is ONE review, not two"


def test_an_edited_sticky_comment_from_a_previous_period_still_charges_this_one():
    """Greptile edits its sticky comment in place, so a re-review keeps the old created_at.

    A run in THIS period on a PR first touched LAST period leaves created_at out of range.
    Placing bot comments by created_at alone would miss it entirely — an undercount, in the
    direction that overruns the cap.
    """
    stale = _bot(THREAD_B, at="2026-07-20T10:00:00Z", cid=7)  # created in the prior period
    stale["updated_at"] = "2026-08-14T10:00:00Z"  # re-reviewed inside this one
    report = _summarise([stale])
    assert report["auto_fire_count"] == 1
    assert report["used"] == 1


def test_a_sticky_comment_untouched_this_period_does_not_charge_it():
    """Positive control: without an in-period edit the same comment must NOT be charged.

    Without this, the previous test would also pass a rule that charged every bot comment
    regardless of when it was written.
    """
    stale = _bot(THREAD_B, at="2026-07-20T10:00:00Z", cid=7)
    stale["updated_at"] = "2026-07-21T10:00:00Z"  # last touched in the prior period
    report = _summarise([stale])
    assert report["auto_fire_count"] == 0
    assert report["used"] == 0


def test_editing_an_old_trigger_comment_does_not_spend_budget():
    """Only bot comments are placed by updated_at — editing a trigger does not re-trigger."""
    stale = _trigger(THREAD_A, at="2026-07-20T10:00:00Z", cid=6)
    stale["updated_at"] = "2026-08-14T10:00:00Z"
    assert _summarise([stale])["used"] == 0


def test_both_greptile_bot_identities_are_recognised():
    """Pinning a single login would silently return 0 forever if the other one posts."""
    for login in ("greptile-apps[bot]", "greptile[bot]"):
        report = _summarise([_bot(THREAD_C, login=login, cid=4)])
        assert report["auto_fire_count"] == 1, login
        assert report["observed_bot_logins"] == [login]


# --- The verdict must be inversion-sensitive -------------------------------------------


@pytest.mark.parametrize(
    ("n_triggers", "limit", "expect_used", "expect_exhausted", "expect_remaining"),
    [
        (3, 5, 3, False, 2),
        (4, 5, 4, False, 1),
        (5, 5, 5, True, 0),  # boundary: used == limit is EXHAUSTED, not available
        (6, 5, 6, True, 0),  # over-run clamps remaining at 0, never negative
    ],
)
def test_exhausted_verdict_at_and_around_the_boundary(
    n_triggers, limit, expect_used, expect_exhausted, expect_remaining
):
    """Asserts both directions of the verdict, so swapping the branches fails the suite."""
    comments = [_trigger(f"{THREAD_A}/{i}", cid=i) for i in range(n_triggers)]
    report = _summarise(comments, limit=limit)
    assert report["used"] == expect_used
    assert report["exhausted"] is expect_exhausted
    assert report["remaining"] == expect_remaining


def test_asymmetric_mix_charges_triggers_and_auto_fires_together():
    """3 triggers + 1 unrequested review against a limit of 5 -> 4 used, 1 left.

    Deliberately asymmetric so that transposing the two terms changes the answer.
    """
    report = _summarise(
        [
            _trigger(THREAD_A, cid=1),
            _trigger(THREAD_B, cid=2),
            _trigger(THREAD_C, cid=3),
            _bot(THREAD_A, cid=4),  # matched -> not charged again
            _bot("https://api.github.com/repos/bioedca/tbox-finder/issues/999", cid=5),
        ]
    )
    assert (report["trigger_count"], report["auto_fire_count"]) == (3, 1)
    assert report["used"] == 4
    assert report["remaining"] == 1
    assert report["exhausted"] is False


# --- Fail-closed -----------------------------------------------------------------------


def test_a_comment_without_a_timestamp_refuses_rather_than_being_dropped():
    broken = _trigger(THREAD_A)
    del broken["created_at"]
    with pytest.raises(mod.BudgetError):
        _summarise([broken])
    # Positive control: the same comment WITH its timestamp counts normally.
    assert _summarise([_trigger(THREAD_A)])["used"] == 1


def test_pagination_truncation_raises_instead_of_undercounting(monkeypatch):
    """A capped page-walk must refuse, not return a short list read as 'few reviews used'."""
    page = (200, b"[]", {"Link": '<https://api.github.com/next>; rel="next"'})
    monkeypatch.setattr(mod, "_get", lambda url, token: page)
    with pytest.raises(mod.BudgetError, match="pagination"):
        mod.fetch_issue_comments("o/r", P_START, None)


@pytest.mark.parametrize("status", [0, 401, 403, 404, 429, 500])
def test_transport_and_http_failures_refuse_rather_than_reporting_zero(monkeypatch, status):
    monkeypatch.setattr(mod, "_get", lambda url, token: (status, b"nope", {}))
    with pytest.raises(mod.BudgetError):
        mod.fetch_issue_comments("o/r", P_START, None)
    # Positive control: a clean 200 with no next-link returns normally.
    monkeypatch.setattr(mod, "_get", lambda url, token: (200, b"[]", {}))
    assert mod.fetch_issue_comments("o/r", P_START, None) == []


def test_next_link_follows_only_rel_next():
    """Link parsing must select rel="next" and ignore every other relation."""
    assert mod._next_link({}) is None
    assert mod._next_link({"Link": '<https://x/2>; rel="next", <https://x/9>; rel="last"'}) == (
        "https://x/2"
    )
    # rel="last" alone must NOT be followed as if it were the next page.
    assert mod._next_link({"Link": '<https://x/9>; rel="last"'}) is None


def test_non_array_payload_refuses(monkeypatch):
    """A 200 carrying an error object must not be iterated as if it were comments."""
    monkeypatch.setattr(mod, "_get", lambda url, token: (200, b'{"message":"Not Found"}', {}))
    with pytest.raises(mod.BudgetError, match="JSON array"):
        mod.fetch_issue_comments("o/r", P_START, None)
    # Positive control: a real array on the same path returns normally.
    monkeypatch.setattr(mod, "_get", lambda url, token: (200, b"[]", {}))
    assert mod.fetch_issue_comments("o/r", P_START, None) == []


def test_unparseable_json_refuses(monkeypatch):
    monkeypatch.setattr(mod, "_get", lambda url, token: (200, b"not json at all", {}))
    with pytest.raises(mod.BudgetError, match="unparseable JSON"):
        mod.fetch_issue_comments("o/r", P_START, None)


@pytest.mark.parametrize("bad", ["not-a-date", "2026-13-01", "", "2026/08/03"])
def test_malformed_anchor_fails_closed_rather_than_crashing(bad):
    """Must raise BudgetError, not ValueError: a bare ValueError exits 1, off-contract."""
    with pytest.raises(mod.BudgetError):
        mod.parse_anchor(bad)
    # Positive control: the real anchor still parses.
    assert mod.parse_anchor("2026-08-03").day == 3


@pytest.mark.parametrize("bad", ["garbage", "2026-08-03", "2026-08-03T00:00:00+00:00", None])
def test_malformed_timestamp_fails_closed_rather_than_crashing(bad):
    with pytest.raises(mod.BudgetError):
        mod._parse_ts(bad)
    # Positive control: the GitHub wire format still parses.
    assert mod._parse_ts("2026-08-03T00:00:00Z").day == 3


@pytest.mark.parametrize(
    "argv",
    [
        ["--anchor", "not-a-date"],
        ["--now", "garbage"],
    ],
)
def test_main_maps_bad_arguments_onto_the_fail_closed_exit_code(monkeypatch, capsys, argv):
    """Bad CLI input must exit 3 (could-not-measure), never 1 (uncaught traceback)."""
    monkeypatch.setattr(mod, "fetch_issue_comments", lambda *a, **k: [])
    assert mod.main(argv) == 3
    assert "COULD NOT MEASURE" in capsys.readouterr().err


def test_counts_via_the_repo_wide_comment_feed_not_a_pr_enumeration(monkeypatch):
    """Locks the endpoint choice that makes this counter immune to `state=open`.

    ``GET /repos/{o}/{r}/pulls`` defaults to **state=open**. This repo squash-merges
    within ~15 min (PR 97: opened 23:51Z, merged 00:05Z), so a PR-enumerating counter that
    forgot ``state=all`` would see an empty set and report "budget remaining" forever.
    ``/repos/{o}/{r}/issues/comments`` has no state filter — verified live on 2026-08-03,
    where it returned comments from merged PRs 94-97 — so the trap is structurally absent.
    Asserted on the URL the script actually builds, not on a mocked return value.
    """
    seen: list[str] = []

    def _capture(url, token):
        seen.append(url)
        return (200, b"[]", {})

    monkeypatch.setattr(mod, "_get", _capture)
    mod.fetch_issue_comments("bioedca/tbox-finder", P_START, None)

    assert len(seen) == 1
    assert "/repos/bioedca/tbox-finder/issues/comments" in seen[0]
    assert "since=2026-08-03T00%3A00%3A00Z" in seen[0]
    # A PR enumeration (the state=open trap) must not be how this count is derived.
    assert "/pulls" not in seen[0]
    assert "state=" not in seen[0]


def test_a_bare_socket_timeout_still_fails_closed(monkeypatch):
    """A read timeout surfaces as bare TimeoutError, which urllib does NOT wrap in URLError.

    Uncaught it escapes the BudgetError path and exits 1, outside the 0/2/3 contract.
    """
    import urllib.request

    def _boom(*a, **k):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(mod.BudgetError):
        mod.fetch_issue_comments("o/r", P_START, None)


def test_http_error_is_not_swallowed_by_the_oserror_catch(monkeypatch):
    """Positive control for catch ordering: HTTPError < URLError < OSError.

    If the OSError arm were placed first it would swallow both, and every HTTP status
    would collapse to the status-0 transport case.
    """
    import urllib.error
    import urllib.request

    def _raise_http(*a, **k):
        raise urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise_http)
    code, _body, _headers = mod._get("http://x", None)
    assert code == 404, "a real HTTP status must survive, not be flattened to 0"


@pytest.mark.parametrize("bad", ["0", "-1", "notanint"])
def test_a_nonsensical_limit_is_a_usage_error_not_a_spent_budget(bad, monkeypatch):
    """`--limit 0` must not report exit 2 ("the month's budget is spent") on an empty period."""
    monkeypatch.setattr(mod, "fetch_issue_comments", lambda *a, **k: [])
    assert mod.main(["--limit", bad]) == mod.USAGE_ERROR_EXIT
    # Positive control: a sane limit on the same empty period is "available", not an error.
    assert mod.main(["--limit", "16"]) == 0


@pytest.mark.parametrize("argv", [["--bogus-flag"], ["--limit", "notanint"]])
def test_usage_errors_do_not_masquerade_as_budget_exhausted(argv, capsys):
    """argparse exits 2 on a usage error — the same code this script means by "exhausted".

    A mistyped flag must not read as "the month's budget is spent".
    """
    rc = mod.main(argv)
    assert rc == mod.USAGE_ERROR_EXIT
    assert rc not in (0, 2, 3), "usage errors must sit outside the budget contract"


def test_help_still_exits_zero(capsys):
    """Positive control: the SystemExit remap must not turn --help into an error."""
    assert mod.main(["--help"]) == 0


def test_main_exit_codes_are_distinct(monkeypatch, capsys):
    """0 available / 2 exhausted / 3 unmeasurable must never collapse into each other."""
    monkeypatch.setattr(mod, "fetch_issue_comments", lambda *a, **k: [_trigger(THREAD_A)])
    assert mod.main(["--now", "2026-08-10T00:00:00Z", "--limit", "5"]) == 0
    assert mod.main(["--now", "2026-08-10T00:00:00Z", "--limit", "1"]) == 2

    def _boom(*a, **k):
        raise mod.BudgetError("simulated outage")

    monkeypatch.setattr(mod, "fetch_issue_comments", _boom)
    assert mod.main(["--now", "2026-08-10T00:00:00Z"]) == 3
    err = capsys.readouterr().err
    assert "COULD NOT MEASURE" in err
    # The unmeasured path must not print a remaining-budget number at all.
    assert "remaining" not in err.lower()
