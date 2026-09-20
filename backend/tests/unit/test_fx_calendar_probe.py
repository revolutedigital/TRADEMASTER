"""Tests for the economic-calendar probe: time zones, page parsers, coverage and the M1 check.

No network is used: every page is a small fixture, and the fetcher gets a fake opener.
"""

import io
import urllib.request
import zipfile
from datetime import date
from datetime import time as clock_time
from pathlib import Path

import pandas as pd
import pytest

from scripts.research import fx_calendar_probe as cal

NEW_YORK = "America/New_York"
BERLIN = "Europe/Berlin"
LONDON = "Europe/London"


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


# --- time zones -------------------------------------------------------------------------------


def test_us_release_times_follow_the_american_daylight_saving_switch() -> None:
    assert cal.to_utc(date(2024, 8, 2), clock_time(8, 30), NEW_YORK) == utc("2024-08-02 12:30")
    assert cal.to_utc(date(2024, 1, 5), clock_time(8, 30), NEW_YORK) == utc("2024-01-05 13:30")
    # The US switched on 2024-03-10: NFP the Friday before is on winter time, the one after is not.
    assert cal.to_utc(date(2024, 3, 8), clock_time(8, 30), NEW_YORK) == utc("2024-03-08 13:30")
    assert cal.to_utc(date(2024, 4, 5), clock_time(8, 30), NEW_YORK) == utc("2024-04-05 12:30")


def test_european_release_times_follow_the_european_switch_not_the_american_one() -> None:
    assert cal.to_utc(date(2024, 9, 12), clock_time(14, 15), BERLIN) == utc("2024-09-12 12:15")
    # The European switch was 2024-03-31: the ECB decision of 03-07 is before it, 04-11 after it.
    assert cal.to_utc(date(2024, 3, 7), clock_time(14, 15), BERLIN) == utc("2024-03-07 13:15")
    assert cal.to_utc(date(2024, 4, 11), clock_time(14, 15), BERLIN) == utc("2024-04-11 12:15")
    assert cal.to_utc(date(2019, 10, 24), clock_time(13, 45), BERLIN) == utc("2019-10-24 11:45")
    assert cal.to_utc(date(2019, 12, 12), clock_time(13, 45), BERLIN) == utc("2019-12-12 12:45")


def test_the_weeks_where_the_two_calendars_disagree_give_different_offsets_per_region() -> None:
    # 2019-03-20: the US is already on summer time, Europe and the UK are not.
    assert cal.to_utc(date(2019, 3, 20), clock_time(14, 0), NEW_YORK) == utc("2019-03-20 18:00")
    assert cal.to_utc(date(2019, 3, 21), clock_time(12, 0), LONDON) == utc("2019-03-21 12:00")
    assert cal.to_utc(date(2019, 3, 7), clock_time(13, 45), BERLIN) == utc("2019-03-07 12:45")
    # 2019-10-30: Europe is back on winter time, the US is still on summer time.
    assert cal.to_utc(date(2019, 10, 30), clock_time(14, 0), NEW_YORK) == utc("2019-10-30 18:00")
    assert cal.to_utc(date(2019, 11, 7), clock_time(12, 0), LONDON) == utc("2019-11-07 12:00")


def test_bank_of_england_noon_is_one_hour_earlier_in_utc_during_british_summer_time() -> None:
    assert cal.to_utc(date(2024, 9, 19), clock_time(12, 0), LONDON) == utc("2024-09-19 11:00")
    assert cal.to_utc(date(2024, 2, 1), clock_time(12, 0), LONDON) == utc("2024-02-01 12:00")


def test_a_time_that_does_not_exist_or_happens_twice_is_an_error() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        cal.to_utc(date(2024, 3, 10), clock_time(2, 30), NEW_YORK)
    with pytest.raises(ValueError, match="ambiguous"):
        cal.to_utc(date(2024, 11, 3), clock_time(1, 30), NEW_YORK)


# --- BLS --------------------------------------------------------------------------------------


def bls_row(day: str, hour: str, description: str) -> str:
    return (
        f'<tr><td class="date-cell"><p>{day}</p></td>\n<td class="time-cell"><p>{hour}</p></td>\n'
        f'<td class="desc-cell"><p>{description}</p></td></tr>'
    )


BLS_PAGE = (
    "<table>"
    + "".join(
        [
            bls_row(
                "Friday, March 08, 2024",
                "08:30 AM",
                "<strong>Employment Situation</strong> for February 2024",
            ),
            bls_row(
                "Tuesday, March 12, 2024",
                "08:30 AM",
                "<strong>Consumer Price Index</strong> for February 2024",
            ),
            bls_row(
                "Wednesday, March 20, 2024",
                "10:00 AM",
                "<strong>Employment Situation of Veterans</strong> for Annual 2023",
            ),
            bls_row(
                "Wednesday, March 06, 2024",
                "10:00 AM",
                "<strong>Job Openings and Labor Turnover Survey</strong> for January 2024",
            ),
            bls_row(
                "Friday, April 5, 2024",
                "08:30 AM",
                "<strong>Employment Situation</strong> for March 2024",
            ),
        ]
    )
    + "</table>"
)


def test_bls_schedule_yields_nfp_and_cpi_with_their_published_time_only() -> None:
    events = cal.parse_bls_schedule(BLS_PAGE, "bls")
    assert [(e.event, e.day, e.local_time) for e in events] == [
        ("NFP", date(2024, 3, 8), clock_time(8, 30)),
        ("CPI US", date(2024, 3, 12), clock_time(8, 30)),
        ("NFP", date(2024, 4, 5), clock_time(8, 30)),
    ]


def test_a_bls_row_whose_weekday_disagrees_with_the_date_is_rejected() -> None:
    page = bls_row(
        "Monday, March 08, 2024",
        "08:30 AM",
        "<strong>Employment Situation</strong> for February 2024",
    )
    with pytest.raises(ValueError, match="weekday"):
        cal.parse_bls_schedule(page, "bls")


def test_a_bls_time_in_an_unexpected_format_is_rejected() -> None:
    page = bls_row(
        "Friday, March 08, 2024", "8:30", "<strong>Employment Situation</strong> for February 2024"
    )
    with pytest.raises(ValueError, match="release time"):
        cal.parse_bls_schedule(page, "bls")


# --- FOMC -------------------------------------------------------------------------------------


def statement_page(day: str, release: str) -> str:
    return (
        f"<p>{day}</p><h3>Federal Reserve issues FOMC statement</h3><p>For release at {release}</p>"
    )


def test_a_fomc_statement_gives_its_release_day_and_time() -> None:
    assert cal.parse_fomc_statement(statement_page("September 18, 2024", "2:00 p.m. EDT")) == (
        date(2024, 9, 18),
        clock_time(14, 0),
    )
    assert cal.parse_fomc_statement(statement_page("March 03, 2020", "10:00 a.m. EST")) == (
        date(2020, 3, 3),
        clock_time(10, 0),
    )
    assert cal.parse_fomc_statement(statement_page("March 15, 2020", "5:00 p.m. EDT")) == (
        date(2020, 3, 15),
        clock_time(17, 0),
    )


def test_a_fomc_zone_label_that_contradicts_the_new_york_calendar_is_an_error() -> None:
    with pytest.raises(ValueError, match="New York is on"):
        cal.parse_fomc_statement(statement_page("December 18, 2024", "2:00 p.m. EDT"))


def test_a_statement_page_without_a_release_time_gives_none() -> None:
    assert cal.parse_fomc_statement("<p>Federal Reserve issues FOMC statement</p>") is None


FOMC_HISTORICAL = """
<h5 class="panel-heading panel-heading--shaded">January 28-29 Meeting - 2020</h5>
<a href="/newsevents/pressreleases/monetary20200129a.htm">Statement</a>
<h5 class="panel-heading panel-heading--shaded">March 2 (unscheduled) Meeting - 2020</h5>
<a href="/newsevents/pressreleases/monetary20200303a.htm">Statement</a>
<h5 class="panel-heading panel-heading--shaded">March 17-18 (cancelled) Meeting - 2020</h5>
<h5 class="panel-heading panel-heading--shaded">March 19 (notation vote) - 2020</h5>
<a href="/newsevents/pressreleases/monetary20200319a.htm">Statement</a>
<h5 class="panel-heading panel-heading--shaded">October 4 (unscheduled) - 2019</h5>
<a href="/newsevents/pressreleases/monetary20191011a.htm">Statement</a>
"""


def test_historical_fomc_pages_keep_meetings_and_skip_votes_cancellations_and_calls() -> None:
    assert cal.fomc_links_historical(FOMC_HISTORICAL) == [
        "/newsevents/pressreleases/monetary20200129a.htm",
        "/newsevents/pressreleases/monetary20200303a.htm",
    ]


def test_the_calendar_page_lists_each_statement_once_and_skips_side_documents() -> None:
    page = (
        '<a id="1">2021 FOMC Meetings</a><a id="2">2022 FOMC Meetings</a>'
        '<a href="/newsevents/pressreleases/monetary20210127a.htm">HTML</a>'
        '<a href="/newsevents/pressreleases/monetary20210127a1.htm">Implementation Note</a>'
        '<a href="/newsevents/pressreleases/monetary20210127b.htm">Longer-Run Goals</a>'
        '<a href="/newsevents/pressreleases/monetary20250822a.htm">Statement on Goals</a>'
        '<a href="/newsevents/pressreleases/monetary20210127a.htm">HTML</a>'
    )
    assert cal.fomc_links_calendar(page) == ["/newsevents/pressreleases/monetary20210127a.htm"]


def test_the_calendar_page_names_the_years_it_covers() -> None:
    page = "<a id='1'>2021 FOMC Meetings</a><a id='2'>2022 FOMC Meetings</a>"
    assert cal.fomc_calendar_years(page) == {2021, 2022}


# --- ECB --------------------------------------------------------------------------------------


def ecb_entry(day: str, path: str, title: str) -> str:
    return (
        f'<dt isoDate="{day}"><div class="date">{day}</div></dt>'
        f'<dd><div class="title"><a href="/press/pr/date/2024/html/{path}.en.html"  >{title}</a>'
        "</div></dd>\n"
    )


ECB_LIST = (
    ecb_entry("2024-09-12", "ecb.mp240912~aa", "Monetary policy decisions")
    + ecb_entry("2024-09-12", "ecb.is240912~bb", "Monetary policy statement (with Q&amp;A)")
    + ecb_entry("2024-10-10", "ecb.ac241010~cc", "Meeting of 11-12 September 2024")
    + ecb_entry("2024-10-17", "ecb.mp241017~dd", "Monetary policy decisions")
)


def test_the_ecb_list_keeps_only_the_monetary_policy_decision_press_releases() -> None:
    assert cal.parse_ecb_decision_list(ECB_LIST) == [
        (date(2024, 9, 12), "/press/pr/date/2024/html/ecb.mp240912~aa.en.html"),
        (date(2024, 10, 17), "/press/pr/date/2024/html/ecb.mp241017~dd.en.html"),
    ]


def test_the_press_conference_time_is_read_from_the_press_release() -> None:
    page = "<p>The President will comment at a press conference starting at 14:45 CET today</p>"
    assert cal.parse_ecb_press_conference(page) == clock_time(14, 45)
    assert cal.parse_ecb_press_conference("<p>nothing here</p>") is None


def test_the_ecb_decision_time_depends_on_the_era_of_the_press_conference_time() -> None:
    assert cal.ecb_decision_time(date(2022, 6, 9), clock_time(14, 30)) == clock_time(13, 45)
    assert cal.ecb_decision_time(date(2022, 7, 21), clock_time(14, 45)) == clock_time(14, 15)
    assert cal.ecb_decision_time(date(2024, 9, 12), clock_time(14, 45)) == clock_time(14, 15)


def test_an_ecb_press_release_that_contradicts_its_era_or_has_no_time_is_not_trusted() -> None:
    assert cal.ecb_decision_time(date(2022, 6, 9), clock_time(14, 45)) is None
    assert cal.ecb_decision_time(date(2024, 9, 12), clock_time(14, 30)) is None
    assert cal.ecb_decision_time(date(2024, 9, 12), clock_time(15, 0)) is None
    assert cal.ecb_decision_time(date(2024, 9, 12), None) is None


def test_the_ecb_timing_notice_must_still_state_both_changes() -> None:
    notice = (
        "Starting from 21 July, monetary policy decisions will be published at 14:15 CET "
        "(instead of 13:45). The press conferences will begin at 14:45 CET (instead of 14:30)."
    )
    assert cal.ecb_notice_confirms_new_times(f"<p>{notice}</p>")
    assert not cal.ecb_notice_confirms_new_times("<p>decisions at 14:15 CET</p>")


# --- Bank of England --------------------------------------------------------------------------


def excel_serial(day: date) -> int:
    return (day - date(1899, 12, 30)).days


def workbook(days: list[date], first_sheet: str = "Bank Rate Decisions") -> bytes:
    cells = "".join(
        f'<row r="{n}"><c r="B{n}" s="10"><v>{excel_serial(d)}</v></c></row>'
        for n, d in enumerate(days, 12)
    )
    cells += '<row r="6"><c r="B6" s="4" t="s"><v>38</v></c></row>'  # a text cell in column B
    cells += '<row r="7"><c r="C7"><v>50000</v></c></row>'  # a number in another column
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook><sheets><sheet name="{first_sheet}" sheetId="1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml", f"<worksheet><sheetData>{cells}</sheetData></worksheet>"
        )
    return buffer.getvalue()


def test_boe_dates_come_from_the_date_column_and_ignore_text_and_other_columns() -> None:
    days = [date(1997, 6, 6), date(2024, 9, 19), date(2024, 11, 7)]
    assert cal.parse_boe_decision_dates(workbook(days)) == days


def test_a_workbook_whose_first_sheet_changed_is_rejected() -> None:
    with pytest.raises(ValueError, match="Bank Rate Decisions"):
        cal.parse_boe_decision_dates(workbook([date(2024, 9, 19)], first_sheet="Other"))


def test_boe_noon_is_only_used_while_the_bank_still_says_it() -> None:
    sentence = (
        "We publish the MPC&rsquo;s decision with the minutes of the meetings at 12 noon on the "
        "day of the announcement, which is usually a Thursday"
    )
    assert cal.boe_states_noon(f"<p>{sentence}</p>")
    assert not cal.boe_states_noon("<p>the minutes of the meetings at 12:30 on the day</p>")


# --- Bank of Japan ----------------------------------------------------------------------------

BOJ_PAGE = """
<table>
<caption class="non-caption">Table : 2026</caption>
<tbody>
<tr><td><a href="k260123a.pdf">Jan. 22 (Thurs.), 23 (Fri.) [PDF 171KB]</a></td><td>-</td></tr>
<tr><td>June 15 (Mon.), 16 (Tues.)</td><td>-</td></tr>
<tr><td>Apr. 30 (Thurs.), May 1 (Fri.)</td><td>-</td></tr>
<tr><td>Sept. 17 (Thurs.), 18 (Fri.)</td><td>-</td></tr>
<tr><td>-</td><td>-</td></tr>
<tr></tr>
</tbody>
</table>
<table><caption>Table : 2020</caption><tr><td>Mar. 16 (Mon.) [PDF 31KB]</td></tr></table>
<table><caption>Not a schedule</caption><tr><td>Jan. 1 (Mon.)</td></tr></table>
"""


def test_the_boj_schedule_gives_the_last_day_of_each_meeting_and_no_time() -> None:
    events = cal.parse_boj_schedule(BOJ_PAGE, "boj")
    assert [(e.day, e.local_time) for e in events] == [
        (date(2026, 1, 23), None),
        (date(2026, 6, 16), None),
        (date(2026, 5, 1), None),
        (date(2026, 9, 18), None),
        (date(2020, 3, 16), None),
    ]


# --- normalisation and coverage ---------------------------------------------------------------


def raw(event: str, day: date, local_time: clock_time | None = None) -> cal.RawEvent:
    return cal.RawEvent(event, day, local_time, "test")


def test_normalize_converts_to_utc_sorts_dedups_and_separates_events_without_time() -> None:
    events = [
        raw("ECB rate decision", date(2024, 9, 12), clock_time(14, 15)),
        raw("NFP", date(2024, 8, 2), clock_time(8, 30)),
        raw("NFP", date(2024, 8, 2), clock_time(8, 30)),
        raw("BoJ rate decision", date(2024, 9, 20)),
        raw("NFP", date(2018, 12, 7), clock_time(8, 30)),
    ]
    calendar, untimed = cal.normalize(events, date(2019, 1, 1), date(2026, 8, 31))
    assert list(calendar.columns[:4]) == ["timestamp_utc", "currency", "event", "impact"]
    assert list(calendar["timestamp_utc"]) == [utc("2024-08-02 12:30"), utc("2024-09-12 12:15")]
    assert list(calendar["currency"]) == ["USD", "EUR"]
    assert set(calendar["impact"]) == {"high"}
    assert str(calendar["timestamp_utc"].dt.tz) == "UTC"
    assert list(untimed["event"]) == ["BoJ rate decision"]
    assert untimed["date"].iloc[0] == date(2024, 9, 20)


def test_normalize_with_no_events_returns_empty_tables_with_the_right_columns() -> None:
    calendar, untimed = cal.normalize([], date(2019, 1, 1), date(2026, 8, 31))
    assert list(calendar.columns) == cal.COLUMNS
    assert calendar.empty and untimed.empty


def test_the_expected_count_is_prorated_by_whole_months_in_a_partial_year() -> None:
    start, end = date(2019, 1, 1), date(2026, 8, 31)
    assert cal.expected_count(12, 2024, start, end) == 12
    assert cal.expected_count(12, 2026, start, end) == 8
    assert cal.expected_count(8, 2026, start, end) == 5
    assert cal.expected_count(8, 2019, start, end) == 8


def test_coverage_counts_matches_up_to_the_expected_and_reports_the_shortfall() -> None:
    start, end = date(2024, 1, 1), date(2025, 12, 31)
    events = [raw("NFP", date(2024, month, 5), clock_time(8, 30)) for month in range(1, 13)]
    events += [raw("NFP", date(2025, month, 5), clock_time(8, 30)) for month in range(1, 12)]
    events += [
        raw("ECB rate decision", date(2024, month, 12), clock_time(14, 15))
        for month in range(1, 10)
    ]
    events += [raw("BoJ rate decision", date(2024, month, 20)) for month in range(1, 9)]
    calendar, untimed = cal.normalize(events, start, end)
    table = cal.coverage_by_year(calendar, untimed, start, end)
    nfp = table[table["event"] == "NFP"].set_index("year")
    assert nfp.loc[2024, ["expected", "found_with_time", "matched"]].tolist() == [12, 12, 12]
    assert nfp.loc[2025, ["expected", "found_with_time", "matched"]].tolist() == [12, 11, 11]
    ecb = table[table["event"] == "ECB rate decision"].set_index("year")
    assert ecb.loc[2024, ["found_with_time", "matched"]].tolist() == [9, 8]  # extras never inflate
    boj = table[table["event"] == "BoJ rate decision"].set_index("year")
    assert boj.loc[2024, ["found_with_time", "found_without_time", "matched"]].tolist() == [0, 8, 0]


def test_the_summary_fails_a_category_below_ninety_percent_and_the_pooled_row_follows() -> None:
    table = pd.DataFrame(
        {
            "event": ["NFP", "NFP", "BoJ rate decision", "BoJ rate decision"],
            "year": [2024, 2025, 2024, 2025],
            "expected": [12, 12, 8, 8],
            "found_with_time": [12, 11, 0, 0],
            "matched": [12, 11, 0, 0],
        }
    )
    summary = cal.summarize_coverage(table)
    assert summary.loc["NFP", "coverage"] == pytest.approx(23 / 24)
    assert bool(summary.loc["NFP", "passes"])
    assert summary.loc["BoJ rate decision", "coverage"] == 0
    assert not bool(summary.loc["BoJ rate decision", "passes"])
    assert summary.loc["ALL", "coverage"] == pytest.approx(23 / 40)
    assert not bool(summary.loc["ALL", "passes"])


# --- M1 check ---------------------------------------------------------------------------------


def quiet_ranges(spike_at: str | None = None, spike: float = 20.0) -> pd.Series:
    index = pd.date_range("2024-08-02 11:00", "2024-08-02 14:00", freq="1min", tz="UTC")
    ranges = pd.Series(1.0, index=index)
    if spike_at:
        ranges.loc[spike_at] = spike
    return ranges


def test_the_reaction_is_the_widest_minute_and_its_offset_is_measured_from_the_expected_time() -> (
    None
):
    expected = utc("2024-08-02 12:30")
    on_time = cal.find_reaction(quiet_ranges("2024-08-02 12:30"), expected)
    assert on_time is not None
    assert (on_time.offset_minutes, on_time.burst_ratio) == (0, 20.0)
    late = cal.find_reaction(quiet_ranges("2024-08-02 12:33"), expected)
    assert late is not None and late.offset_minutes == 3
    wrong_hour = cal.find_reaction(quiet_ranges("2024-08-02 13:30"), expected)
    assert wrong_hour is not None and wrong_hour.burst_ratio == 1.0


def test_no_reaction_is_reported_without_bars_around_the_release() -> None:
    assert cal.find_reaction(quiet_ranges(), utc("2024-08-05 12:30")) is None


def test_reaction_summary_counts_only_clear_bursts_and_on_minute_peaks() -> None:
    calendar = pd.DataFrame(
        {
            "timestamp_utc": [utc("2024-08-02 12:30")] * 2,
            "currency": ["USD", "USD"],
            "event": ["NFP", "NFP"],
            "impact": ["high", "high"],
            "source": ["t", "t"],
        }
    )
    on_time = cal.reaction_table(calendar, {"EURUSD": quiet_ranges("2024-08-02 12:30")})
    flat = cal.reaction_table(calendar.iloc[:1], {"EURUSD": quiet_ranges()})
    summary = cal.summarize_reactions(pd.concat([on_time, flat], ignore_index=True))
    assert summary.loc["NFP", "events"] == 3
    assert summary.loc["NFP", "clear_burst"] == 2
    assert summary.loc["NFP", "on_expected_minute"] == 2
    assert summary.loc["NFP", "within_one_minute"] == 2


def test_a_stronger_burst_an_hour_away_is_flagged_as_a_possible_wrong_offset() -> None:
    calendar = pd.DataFrame(
        {
            "timestamp_utc": [utc("2024-08-02 12:30")],
            "currency": ["USD"],
            "event": ["NFP"],
            "impact": ["high"],
            "source": ["t"],
        }
    )
    table = cal.reaction_table(calendar, {"EURUSD": quiet_ranges("2024-08-02 13:30")})
    assert table["shifted_burst_ratio"].iloc[0] == 20.0
    summary = cal.summarize_reactions(table)
    assert summary.loc["NFP", "shifted_hour_stronger"] == 1
    assert summary.loc["NFP", "clear_burst"] == 0


def test_minute_ranges_read_every_m1_file_of_the_pair(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "bid_high": [1.10, 1.20],
            "bid_low": [1.00, 1.10],
            "ask_high": [1.12, 1.22],
            "ask_low": [1.02, 1.12],
        },
        index=index,
    )
    bars.iloc[:1].to_parquet(tmp_path / "EURUSD_M1_a.parquet")
    bars.iloc[1:].to_parquet(tmp_path / "EURUSD_M1_b.parquet")
    ranges = cal.load_minute_ranges(tmp_path, "EURUSD")
    assert ranges.tolist() == pytest.approx([0.10, 0.10])
    with pytest.raises(FileNotFoundError):
        cal.load_minute_ranges(tmp_path, "GBPUSD")


# --- fetching ---------------------------------------------------------------------------------


class FakeSite:
    """Answers requests from a table of url -> list of (status, body), recording each call."""

    def __init__(self, answers: dict[str, list[tuple[int, bytes]]]) -> None:
        self.answers = {url: list(items) for url, items in answers.items()}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: urllib.request.Request) -> tuple[int, bytes]:
        self.calls.append((request.full_url, request.get_header("User-agent")))
        queue = self.answers.get(request.full_url, [(404, b"missing")])
        return queue.pop(0) if len(queue) > 1 else queue[0]


def make_fetcher(
    tmp_path: Path, site: FakeSite, **options: object
) -> tuple[cal.Fetcher, list[float]]:
    pauses: list[float] = []
    fetcher = cal.Fetcher(tmp_path, opener=site, sleep=pauses.append)
    for name, value in options.items():
        setattr(fetcher, name, value)
    return fetcher, pauses


def test_a_page_is_downloaded_once_then_served_from_the_cache(tmp_path: Path) -> None:
    site = FakeSite({"https://example.gov/a.htm": [(200, b"page")]})
    fetcher, pauses = make_fetcher(tmp_path, site)
    assert fetcher.get("https://example.gov/a.htm") == b"page"
    calls_after_first = len(site.calls)
    assert fetcher.get_text("https://example.gov/a.htm") == "page"
    assert len(site.calls) == calls_after_first  # robots.txt and page fetched once, not again
    assert site.calls[0][0] == "https://example.gov/robots.txt"
    assert pauses[:2] == [1.5, 1.5]  # a pause before every network call


def test_refresh_downloads_again_even_when_cached(tmp_path: Path) -> None:
    site = FakeSite({"https://example.gov/a.htm": [(200, b"old"), (200, b"new")]})
    fetcher, _ = make_fetcher(tmp_path, site)
    assert fetcher.get("https://example.gov/a.htm") == b"old"
    fetcher.refresh = True
    assert fetcher.get("https://example.gov/a.htm") == b"new"


def test_robots_txt_can_forbid_a_path(tmp_path: Path) -> None:
    site = FakeSite(
        {
            "https://example.gov/robots.txt": [(200, b"User-agent: *\nDisallow: /private\n")],
            "https://example.gov/private/x.htm": [(200, b"secret")],
        }
    )
    fetcher, _ = make_fetcher(tmp_path, site)
    with pytest.raises(cal.FetchError, match="robots.txt forbids"):
        fetcher.get("https://example.gov/private/x.htm")
    assert all(url.endswith("robots.txt") for url, _ in site.calls)


def test_the_user_agent_names_the_probe_and_carries_the_contact_when_given(tmp_path: Path) -> None:
    site = FakeSite({"https://example.gov/a.htm": [(200, b"page")]})
    fetcher, _ = make_fetcher(tmp_path, site, contact="research@example.org")
    fetcher.get("https://example.gov/a.htm")
    assert {agent for _, agent in site.calls} == {
        "trademaster-calendar-probe/1.0 (personal research, low volume; research@example.org)"
    }


def test_a_403_explains_the_bls_contact_rule_and_is_not_retried(tmp_path: Path) -> None:
    site = FakeSite({"https://www.bls.gov/x.htm": [(403, b"Access Denied")]})
    fetcher, _ = make_fetcher(tmp_path, site)
    with pytest.raises(cal.FetchError, match="--contact"):
        fetcher.get("https://www.bls.gov/x.htm")
    assert [url for url, _ in site.calls].count("https://www.bls.gov/x.htm") == 1


def test_server_errors_are_retried_with_growing_pauses_then_reported(tmp_path: Path) -> None:
    site = FakeSite({"https://example.gov/a.htm": [(503, b"busy")]})
    fetcher, pauses = make_fetcher(tmp_path, site, retries=3)
    with pytest.raises(cal.FetchError, match="status 503"):
        fetcher.get("https://example.gov/a.htm")
    assert [url for url, _ in site.calls].count("https://example.gov/a.htm") == 3
    assert [pause for pause in pauses if pause > 1.5] == [3.0, 6.0]  # backoff doubles


def test_a_page_that_is_not_found_is_an_error_and_plain_http_is_refused(tmp_path: Path) -> None:
    fetcher, _ = make_fetcher(tmp_path, FakeSite({}))
    with pytest.raises(cal.FetchError, match="404"):
        fetcher.get("https://example.gov/missing.htm")
    with pytest.raises(ValueError, match="https"):
        fetcher.get("http://example.gov/a.htm")


def test_cache_names_are_safe_file_names() -> None:
    name = cal.cache_name(
        "https://www.ecb.europa.eu/press/govcdec/mopo/2024/html/index_include.en.html"
    )
    assert "/" not in name and name.endswith("index_include.en.html")
    assert cal.cache_name("https://a.gov/p?x=1&y=2") != cal.cache_name("https://a.gov/p?x=1")


# --- collection and command line --------------------------------------------------------------


class StubFetcher(cal.Fetcher):
    """A fetcher that serves pages from a dict and never touches the network."""

    def __init__(self, pages: dict[str, bytes]) -> None:
        super().__init__(Path("unused"))
        self.pages = pages

    def get(self, url: str) -> bytes:
        if url not in self.pages:
            raise cal.FetchError(f"unexpected url {url}")
        return self.pages[url]


WINDOW = (date(2019, 1, 1), date(2026, 8, 31))


def test_fomc_collection_reads_each_statement_and_keeps_the_link_day_when_the_time_is_missing() -> (
    None
):
    calendar_page = (
        "<a>2021 FOMC Meetings</a>"
        '<a href="/newsevents/pressreleases/monetary20210127a.htm">HTML</a>'
        '<a href="/newsevents/pressreleases/monetary20210317a.htm">HTML</a>'
        '<a href="/newsevents/pressreleases/monetary20270127a.htm">HTML</a>'
    )
    base = "https://www.federalreserve.gov"
    fetcher = StubFetcher(
        {
            cal.FOMC_CALENDAR: calendar_page.encode(),
            base + "/monetarypolicy/fomchistorical2019.htm": FOMC_HISTORICAL.encode(),
            base + "/monetarypolicy/fomchistorical2020.htm": FOMC_HISTORICAL.encode(),
            base + "/monetarypolicy/fomchistorical2022.htm": b"",
            base + "/monetarypolicy/fomchistorical2023.htm": b"",
            base + "/monetarypolicy/fomchistorical2024.htm": b"",
            base + "/monetarypolicy/fomchistorical2025.htm": b"",
            base + "/monetarypolicy/fomchistorical2026.htm": b"",
            base + "/newsevents/pressreleases/monetary20200129a.htm": statement_page(
                "January 29, 2020", "2:00 p.m. EST"
            ).encode(),
            base + "/newsevents/pressreleases/monetary20200303a.htm": statement_page(
                "March 03, 2020", "10:00 a.m. EST"
            ).encode(),
            base + "/newsevents/pressreleases/monetary20210127a.htm": statement_page(
                "January 27, 2021", "2:00 p.m. EST"
            ).encode(),
            base + "/newsevents/pressreleases/monetary20210317a.htm": b"<p>no time here</p>",
        }
    )
    events = cal.collect_fomc(fetcher, *WINDOW)
    assert [(e.day, e.local_time) for e in events] == [
        (date(2021, 1, 27), clock_time(14, 0)),
        (date(2021, 3, 17), None),
        (date(2020, 1, 29), clock_time(14, 0)),
        (date(2020, 3, 3), clock_time(10, 0)),
    ]


def test_ecb_collection_maps_each_press_conference_time_to_the_decision_time() -> None:
    notice = (
        "published at 14:15 CET (instead of 13:45). The press conferences will begin at 14:45 CET "
        "(instead of 14:30)."
    )
    pages = {cal.ECB_TIMING_NOTICE: f"<p>{notice}</p>".encode()}
    for year in range(2019, 2027):
        pages[cal.ECB_DECISION_LIST.format(year=year)] = b""
    pages[cal.ECB_DECISION_LIST.format(year=2024)] = ECB_LIST.encode()
    pages[cal.ECB_BASE + "/press/pr/date/2024/html/ecb.mp240912~aa.en.html"] = (
        b"<p>press conference starting at 14:45 CET today</p>"
    )
    pages[cal.ECB_BASE + "/press/pr/date/2024/html/ecb.mp241017~dd.en.html"] = (
        b"<p>press conference starting at 14:30 CET today</p>"  # contradicts its era
    )
    events = cal.collect_ecb(StubFetcher(pages), *WINDOW)
    assert [(e.day, e.local_time) for e in events] == [
        (date(2024, 9, 12), clock_time(14, 15)),
        (date(2024, 10, 17), None),
    ]


def test_ecb_collection_stops_when_the_timing_notice_no_longer_confirms_the_change() -> None:
    fetcher = StubFetcher({cal.ECB_TIMING_NOTICE: b"<p>something else</p>"})
    with pytest.raises(cal.FetchError, match="no longer confirms"):
        cal.collect_ecb(fetcher, *WINDOW)


def test_boe_collection_needs_the_noon_sentence_to_give_events_a_time() -> None:
    noon_page = b"<p>the minutes of the meetings at 12 noon on the day of the announcement</p>"
    days = [date(2018, 12, 20), date(2024, 9, 19), date(2026, 9, 17)]
    pages = {cal.BOE_VOTING_HISTORY: workbook(days), cal.BOE_MONETARY_POLICY_PAGE: noon_page}
    events = cal.collect_boe(StubFetcher(pages), *WINDOW)
    assert [(e.day, e.local_time) for e in events] == [(date(2024, 9, 19), clock_time(12, 0))]
    pages[cal.BOE_MONETARY_POLICY_PAGE] = b"<p>the time changed</p>"
    assert cal.collect_boe(StubFetcher(pages), *WINDOW)[0].local_time is None


def test_main_reports_a_fetch_failure_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*_: object) -> list[cal.RawEvent]:
        raise cal.FetchError("robots.txt forbids something")

    monkeypatch.setattr(cal, "collect_all", refuse)
    assert cal.main(["--workdir", str(tmp_path)]) == 2
    assert "fetch failed: robots.txt forbids something" in capsys.readouterr().out


def test_main_writes_the_calendar_coverage_and_a_failing_exit_code_below_the_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        raw("NFP", date(2024, 8, 2), clock_time(8, 30)),
        raw("BoJ rate decision", date(2024, 9, 20)),
    ]
    monkeypatch.setattr(cal, "collect_all", lambda *_: events)
    code = cal.main(
        ["--workdir", str(tmp_path), "--skip-m1", "--start", "2024-01-01", "--end", "2024-12-31"]
    )
    assert code == 1  # the categories with no events in the window fall below the bar
    calendar = pd.read_csv(tmp_path / "events.csv", parse_dates=["timestamp_utc"])
    assert calendar["timestamp_utc"].iloc[0] == utc("2024-08-02 12:30")
    assert list(pd.read_csv(tmp_path / "events_without_time.csv")["event"]) == ["BoJ rate decision"]
    assert (tmp_path / "coverage_summary.csv").exists()
