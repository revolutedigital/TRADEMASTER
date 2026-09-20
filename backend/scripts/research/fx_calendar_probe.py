"""Research-only probe: a historical high-impact economic calendar built from official sources.

Family F4 (drift after news) only enters the fast-strategy lab if a calendar gives a verifiable
release time for the high-impact events of 2019-01..2026-08. Aggregators (Forex Factory,
Investing.com) forbid copying their history, so the calendar is assembled from the issuers, which
publish it free of charge with attribution:

* NFP and US CPI: the BLS yearly release schedule (date and time of every release).
* FOMC: federalreserve.gov meeting pages; the time is read from each statement ("For release at").
* ECB: the monetary policy decision list; the press conference time is read from each press
  release and mapped to the decision time with the ECB notice of 2022-06-27 (13:45 -> 14:15 CET
  from 2022-07-21). The ECB writes "CET" all year, meaning Frankfurt local time.
* BoE: the MPC voting history workbook (announcement dates); 12:00 London time, checked against
  the sentence on the Bank's monetary policy page.
* BoJ: the MPM schedule. The Bank publishes no time for the policy statement, so those rows are
  written to a separate file instead of being given an invented time.

Every local time is converted to UTC with the IANA zone rules, never with a fixed offset (the US
and Europe switch on different dates). The result can be checked against the market itself: the
minute with the widest range around each release is compared with the expected minute in the M1
bars already in the project.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from html import unescape
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd

WINDOW_START = date(2019, 1, 1)
WINDOW_END = date(2026, 8, 31)
MIN_COVERAGE = 0.90
BURST_RATIO = 5.0
ECB_TIMING_NOTICE = (
    "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.pr220627~73acedf868.en.html"
)
BOE_MONETARY_POLICY_PAGE = "https://www.bankofengland.co.uk/monetary-policy"
BOE_VOTING_HISTORY = (
    "https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-summary-and-minutes/"
    "mpcvoting.xlsx"
)
BOJ_SCHEDULE_PAGES = (
    "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
    "https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm",
)
FED_BASE = "https://www.federalreserve.gov"
FOMC_CALENDAR = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
ECB_DECISION_LIST = "https://www.ecb.europa.eu/press/govcdec/mopo/{year}/html/index_include.en.html"
ECB_BASE = "https://www.ecb.europa.eu"
ECB_TIME_CHANGE = date(2022, 7, 21)
ECB_DECISION_BY_PRESS_CONFERENCE = {
    clock_time(14, 30): clock_time(13, 45),
    clock_time(14, 45): clock_time(14, 15),
}
BLS_SCHEDULE = "https://www.bls.gov/schedule/{year}/home.htm"
BLS_RELEASE_TIME = re.compile(r"^\d{2}:\d{2} [AP]M$")


@dataclass(frozen=True)
class Category:
    """One kind of high-impact event: its currency, nominal frequency and clock."""

    event: str
    currency: str
    per_year: int
    zone: str


CATEGORIES = {
    category.event: category
    for category in (
        Category("NFP", "USD", 12, "America/New_York"),
        Category("CPI US", "USD", 12, "America/New_York"),
        Category("FOMC rate decision", "USD", 8, "America/New_York"),
        Category("ECB rate decision", "EUR", 8, "Europe/Berlin"),
        Category("BoE rate decision", "GBP", 8, "Europe/London"),
        Category("BoJ rate decision", "JPY", 8, "Asia/Tokyo"),
    )
}
PAIR_BY_CURRENCY = {"USD": "EURUSD", "EUR": "EURUSD", "GBP": "GBPUSD", "JPY": "USDJPY"}


@dataclass(frozen=True)
class RawEvent:
    """A release as the issuer states it: local calendar day and, if published, local time."""

    event: str
    day: date
    local_time: clock_time | None
    source: str


# ---------------------------------------------------------------------------------------------
# Time zones
# ---------------------------------------------------------------------------------------------


def to_utc(day: date, local_time: clock_time, zone: str) -> pd.Timestamp:
    """Convert a wall-clock time in an IANA zone to UTC, refusing times that do not exist once.

    Clocks skip an hour in spring (the time never happens) and repeat one in autumn (the time
    happens twice); both would silently give a wrong instant, so they are errors here.
    """
    tz = ZoneInfo(zone)
    naive = datetime.combine(day, local_time)
    first = naive.replace(tzinfo=tz, fold=0)
    if first.astimezone(ZoneInfo("UTC")).astimezone(tz).replace(tzinfo=None) != naive:
        raise ValueError(f"{naive} does not exist in {zone}")
    if first.utcoffset() != naive.replace(tzinfo=tz, fold=1).utcoffset():
        raise ValueError(f"{naive} is ambiguous in {zone}")
    return pd.Timestamp(first.astimezone(ZoneInfo("UTC")))


# ---------------------------------------------------------------------------------------------
# Polite fetching
# ---------------------------------------------------------------------------------------------


class FetchError(RuntimeError):
    """A page could not be retrieved in a way that respects the site's rules."""


Opener = Callable[[urllib.request.Request], tuple[int, bytes]]


def _urlopen(request: urllib.request.Request) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - https only
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def cache_name(url: str) -> str:
    """A readable, filesystem-safe file name for a URL."""
    parts = urlsplit(url)
    return re.sub(r"[^A-Za-z0-9._~-]+", "_", f"{parts.netloc}{parts.path}?{parts.query}").rstrip(
        "?_"
    )


@dataclass
class Fetcher:
    """Downloads each page once, identifies itself, honours robots.txt and pauses between calls."""

    cache_dir: Path
    contact: str | None = None
    pause_seconds: float = 1.5
    retries: int = 3
    refresh: bool = False
    opener: Opener = _urlopen
    sleep: Callable[[float], None] = time.sleep
    _robots: dict[str, urllib.robotparser.RobotFileParser] = field(default_factory=dict)

    @property
    def user_agent(self) -> str:
        suffix = f"; {self.contact}" if self.contact else ""
        return f"trademaster-calendar-probe/1.0 (personal research, low volume{suffix})"

    def get(self, url: str) -> bytes:
        """Return the page body, from the cache when it was downloaded before."""
        if not url.startswith("https://"):
            raise ValueError(f"only https URLs are fetched: {url}")
        target = self.cache_dir / cache_name(url)
        if target.exists() and not self.refresh:
            return target.read_bytes()
        self._require_allowed(url)
        body = self._download(url)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return body

    def get_text(self, url: str) -> str:
        return self.get(url).decode("utf-8", errors="replace")

    def _require_allowed(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.netloc not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            status, body = self._request(f"https://{parts.netloc}/robots.txt")
            if status == 200:
                parser.parse(body.decode("utf-8", errors="replace").splitlines())
            elif 400 <= status < 500:
                parser.parse([])  # no robots.txt: everything is allowed
            else:
                raise FetchError(f"robots.txt of {parts.netloc} answered {status}")
            self._robots[parts.netloc] = parser
        if not self._robots[parts.netloc].can_fetch(self.user_agent, url):
            raise FetchError(f"robots.txt forbids {url}")

    def _request(self, url: str) -> tuple[int, bytes]:
        request = urllib.request.Request(  # noqa: S310 - only https URLs reach this point
            url, headers={"User-Agent": self.user_agent}
        )
        self.sleep(self.pause_seconds)
        return self.opener(request)

    def _download(self, url: str) -> bytes:
        last_problem = ""
        for attempt in range(self.retries):
            try:
                status, body = self._request(url)
            except OSError as error:  # timeouts and resets are transient
                last_problem = repr(error)
            else:
                if status == 200:
                    return body
                if status == 403:
                    raise FetchError(
                        f"{url} answered 403; the BLS blocks robots that do not identify an owner, "
                        "pass --contact with an address or URL where you can be reached"
                    )
                if status < 500 and status != 429:
                    raise FetchError(f"{url} answered {status}")
                last_problem = f"status {status}"
            self.sleep(self.pause_seconds * 2**attempt)
        raise FetchError(f"could not download {url}: {last_problem}")


# ---------------------------------------------------------------------------------------------
# Parsers (pure functions over page text, tested without network)
# ---------------------------------------------------------------------------------------------


def visible_text(html: str) -> str:
    """Page text with scripts, tags and repeated whitespace removed."""
    stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", stripped))).strip()


_BLS_ROW = re.compile(
    r'<td class="date-cell"><p>(.*?)</p></td>\s*<td class="time-cell"><p>(.*?)</p></td>\s*'
    r'<td class="desc-cell"><p>(.*?)</p></td>',
    re.S,
)
_BLS_TITLES = {
    "NFP": re.compile(r"^Employment Situation(?: \(\w\))? for "),
    "CPI US": re.compile(r"^Consumer Price Index(?: \(\w\))? for "),
}


def parse_bls_schedule(html: str, source: str) -> list[RawEvent]:
    """NFP and CPI rows of one BLS yearly schedule page ("Friday, March 08, 2024", "08:30 AM")."""
    events = []
    for raw_day, raw_time, raw_description in _BLS_ROW.findall(html):
        description = visible_text(raw_description)
        for event, title in _BLS_TITLES.items():
            if title.match(description):
                events.append(_bls_event(event, visible_text(raw_day), raw_time.strip(), source))
    return events


def _bls_event(event: str, raw_day: str, raw_time: str, source: str) -> RawEvent:
    parsed_day = datetime.strptime(raw_day, "%A, %B %d, %Y")
    if parsed_day.strftime("%A") != raw_day.split(",")[0]:
        raise ValueError(f"weekday does not match the date: {raw_day}")
    if not BLS_RELEASE_TIME.match(raw_time):
        raise ValueError(f"unexpected BLS release time {raw_time!r} on {raw_day}")
    return RawEvent(
        event, parsed_day.date(), datetime.strptime(raw_time, "%I:%M %p").time(), source
    )


_FOMC_LINK = re.compile(r'href="(/newsevents/pressreleases/monetary(\d{8})a\.htm)"')
_FOMC_CALENDAR_LINK = re.compile(_FOMC_LINK.pattern + r">HTML</a>")
_FOMC_CALENDAR_YEAR = re.compile(r">(\d{4}) FOMC Meetings<")


def fomc_calendar_years(html: str) -> set[int]:
    """Years listed on the main FOMC calendar page (the older ones live on historical pages)."""
    return {int(year) for year in _FOMC_CALENDAR_YEAR.findall(html)}


def fomc_links_calendar(html: str) -> list[str]:
    """Statement links of the main calendar page: one per meeting, in page order.

    Only links labelled "HTML" count: other `a.htm` releases (the 2025-08-22 statement on longer-run
    goals) are documents, not rate decisions.
    """
    return list(dict.fromkeys(match.group(1) for match in _FOMC_CALENDAR_LINK.finditer(html)))


def fomc_links_historical(html: str) -> list[str]:
    """Statement links of a historical page, keeping only rate-decision meetings.

    Notation votes (statements approved without a meeting) and cancelled meetings are skipped;
    unscheduled meetings that issued a statement are kept because they moved rates.
    """
    links = []
    for section in re.split(r'<h5 class="panel-heading', html)[1:]:
        heading = visible_text(section.split("</h5>")[0])
        link = _FOMC_LINK.search(section)
        if link and re.search(r"\bMeeting\b", heading) and "cancelled" not in heading:
            links.append(link.group(1))
    return links


_FED_STATEMENT = re.compile(
    r"(?P<day>[A-Z][a-z]+ \d{1,2}, \d{4}) Federal Reserve issues FOMC statement "
    r"For release at (?P<clock>\d{1,2}:\d{2}) (?P<half>[ap])\.m\. (?P<label>E[SD]T)"
)


def parse_fomc_statement(html: str) -> tuple[date, clock_time] | None:
    """Release day and time of one FOMC statement page, or None when the page has no time.

    The page also names the zone (EST or EDT); it must agree with the New York calendar, which is
    an independent check on the daylight-saving handling.
    """
    match = _FED_STATEMENT.search(visible_text(html))
    if match is None:
        return None
    day = datetime.strptime(match["day"], "%B %d, %Y").date()
    release = datetime.strptime(f"{match['clock']} {match['half']}m", "%I:%M %p").time()
    label = datetime.combine(day, release).replace(tzinfo=ZoneInfo("America/New_York")).tzname()
    if label != match["label"]:
        raise ValueError(f"Fed says {match['label']} but New York is on {label} on {day}")
    return day, release


_ECB_DECISION = re.compile(
    r'<dt isoDate="(\d{4}-\d{2}-\d{2})">(?:(?!<dt).)*?<div class="title"><a href="([^"]+)"[^>]*>'
    r"\s*Monetary policy decisions\s*</a>",
    re.S,
)
_ECB_PRESS_CONFERENCE = re.compile(r"press conference starting at (\d{2}):(\d{2}) CET")


def parse_ecb_decision_list(html: str) -> list[tuple[date, str]]:
    """(day, press release path) of every "Monetary policy decisions" entry of a year list."""
    return [(date.fromisoformat(day), path) for day, path in _ECB_DECISION.findall(html)]


def parse_ecb_press_conference(html: str) -> clock_time | None:
    """Press conference start (Frankfurt local time) stated in a decision press release."""
    match = _ECB_PRESS_CONFERENCE.search(visible_text(html))
    return clock_time(int(match[1]), int(match[2])) if match else None


def ecb_decision_time(day: date, press_conference: clock_time | None) -> clock_time | None:
    """Decision time implied by the press conference time, or None when they do not agree.

    Before 2022-07-21 the decision came at 13:45 for a 14:30 press conference; from that day on it
    comes at 14:15 for 14:45. A press release that contradicts its own era is not trusted.
    """
    if press_conference not in ECB_DECISION_BY_PRESS_CONFERENCE:
        return None
    new_era = press_conference == clock_time(14, 45)
    if new_era != (day >= ECB_TIME_CHANGE):
        return None
    return ECB_DECISION_BY_PRESS_CONFERENCE[press_conference]


def ecb_notice_confirms_new_times(html: str) -> bool:
    """True when the ECB notice of 2022-06-27 still states both timing changes."""
    text = visible_text(html)
    return (
        "published at 14:15 CET (instead of 13:45)" in text
        and "begin at 14:45 CET (instead of 14:30)" in text
    )


def boe_states_noon(html: str) -> bool:
    """True when the Bank's monetary policy page still promises publication at 12 noon."""
    return "the minutes of the meetings at 12 noon on the day of the announcement" in visible_text(
        html
    )


_XLSX_DATE_CELL = re.compile(r'<c r="B\d+"([^>]*)><v>(\d+(?:\.\d+)?)</v>')


def parse_boe_decision_dates(workbook: bytes) -> list[date]:
    """Announcement dates of the "Bank Rate Decisions" sheet of the MPC voting history.

    Column B holds the date as an Excel serial number; text cells (member names) are skipped.
    """
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        first_sheet = re.search(
            r'<sheet [^>]*name="([^"]+)"', archive.read("xl/workbook.xml").decode()
        )
        if first_sheet is None or first_sheet[1] != "Bank Rate Decisions":
            raise ValueError("the first sheet of the MPC workbook is not 'Bank Rate Decisions'")
        sheet = archive.read("xl/worksheets/sheet1.xml").decode()
    serials = [
        float(value)
        for attributes, value in _XLSX_DATE_CELL.findall(sheet)
        if ' t="' not in attributes
    ]
    excel_epoch = date(1899, 12, 30)
    return [
        excel_epoch + timedelta(days=int(serial)) for serial in serials if 35000 < serial < 60000
    ]


_BOJ_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S)
_BOJ_YEAR = re.compile(r"<caption[^>]*>\s*Table : (\d{4})")
_BOJ_DAYS = re.compile(
    r"^([A-Z][a-z]{2,4})\.? (\d{1,2}) \([A-Za-z.]+\)"
    r"(?:, (?:([A-Z][a-z]{2,4})\.? )?(\d{1,2}) \([A-Za-z.]+\))?"
)


def parse_boj_schedule(html: str, source: str) -> list[RawEvent]:
    """Decision days (the last day of each meeting) from the MPM schedule tables, without a time."""
    events = []
    for table in _BOJ_TABLE.findall(html):
        year_match = _BOJ_YEAR.search(table)
        if year_match is None:
            continue
        year = int(year_match[1])
        for row in re.findall(r"<tr>(.*?)</tr>", table, re.S):
            cell = re.search(r"<td[^>]*>(.*?)</td>", row, re.S)
            days = _BOJ_DAYS.match(visible_text(cell[1])) if cell else None
            if days is None:
                continue
            first_month, first_day, last_month, last_day = days.groups()
            month_name = last_month or first_month
            day = int(last_day or first_day)
            month = datetime.strptime(month_name[:3], "%b").month
            events.append(RawEvent("BoJ rate decision", date(year, month, day), None, source))
    return events


# ---------------------------------------------------------------------------------------------
# Collection: from the issuers' pages to raw events
# ---------------------------------------------------------------------------------------------


def _years(start: date, end: date) -> range:
    return range(start.year, end.year + 1)


def collect_bls(fetcher: Fetcher, start: date, end: date) -> list[RawEvent]:
    events: list[RawEvent] = []
    for year in _years(start, end):
        url = BLS_SCHEDULE.format(year=year)
        events += parse_bls_schedule(fetcher.get_text(url), url)
    return events


def collect_fomc(fetcher: Fetcher, start: date, end: date) -> list[RawEvent]:
    calendar_html = fetcher.get_text(FOMC_CALENDAR)
    links = fomc_links_calendar(calendar_html)
    for year in sorted(set(_years(start, end)) - fomc_calendar_years(calendar_html)):
        links += fomc_links_historical(
            fetcher.get_text(f"{FED_BASE}/monetarypolicy/fomchistorical{year}.htm")
        )
    events = []
    for path in dict.fromkeys(links):
        url = FED_BASE + path
        link_day = datetime.strptime(path.split("monetary")[1][:8], "%Y%m%d").date()
        if not start <= link_day <= end:
            continue
        released = parse_fomc_statement(fetcher.get_text(url))
        day, local_time = released if released else (link_day, None)
        events.append(RawEvent("FOMC rate decision", day, local_time, url))
    return events


def collect_ecb(fetcher: Fetcher, start: date, end: date) -> list[RawEvent]:
    if not ecb_notice_confirms_new_times(fetcher.get_text(ECB_TIMING_NOTICE)):
        raise FetchError("the ECB notice no longer confirms the 13:45 -> 14:15 change")
    events = []
    for year in _years(start, end):
        listing = fetcher.get_text(ECB_DECISION_LIST.format(year=year))
        for day, path in parse_ecb_decision_list(listing):
            if start <= day <= end:
                url = ECB_BASE + path
                decision = ecb_decision_time(day, parse_ecb_press_conference(fetcher.get_text(url)))
                events.append(RawEvent("ECB rate decision", day, decision, url))
    return events


def collect_boe(fetcher: Fetcher, start: date, end: date) -> list[RawEvent]:
    noon = (
        clock_time(12, 0) if boe_states_noon(fetcher.get_text(BOE_MONETARY_POLICY_PAGE)) else None
    )
    days = parse_boe_decision_dates(fetcher.get(BOE_VOTING_HISTORY))
    return [
        RawEvent("BoE rate decision", day, noon, BOE_VOTING_HISTORY)
        for day in days
        if start <= day <= end
    ]


def collect_boj(fetcher: Fetcher, start: date, end: date) -> list[RawEvent]:
    events: list[RawEvent] = []
    for url in BOJ_SCHEDULE_PAGES:
        events += parse_boj_schedule(fetcher.get_text(url), url)
    return [event for event in events if start <= event.day <= end]


def collect_all(fetcher: Fetcher, start: date, end: date) -> list[RawEvent]:
    collectors = (collect_bls, collect_fomc, collect_ecb, collect_boe, collect_boj)
    return [event for collect in collectors for event in collect(fetcher, start, end)]


# ---------------------------------------------------------------------------------------------
# Normalisation and coverage
# ---------------------------------------------------------------------------------------------

COLUMNS = ["timestamp_utc", "currency", "event", "impact", "source"]


def normalize(events: list[RawEvent], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw events into the UTC calendar and the events whose time is not published.

    `impact` is "high" by construction: only the six high-impact categories are collected.
    """
    timed, untimed = [], []
    for event in sorted(set(events), key=lambda item: (item.day, item.event)):
        if not start <= event.day <= end:
            continue
        category = CATEGORIES[event.event]
        if event.local_time is None:
            untimed.append(
                {
                    "date": event.day,
                    "currency": category.currency,
                    "event": event.event,
                    "source": event.source,
                }
            )
            continue
        timed.append(
            {
                "timestamp_utc": to_utc(event.day, event.local_time, category.zone),
                "currency": category.currency,
                "event": event.event,
                "impact": "high",
                "source": event.source,
            }
        )
    calendar = pd.DataFrame(timed, columns=COLUMNS).drop_duplicates(["timestamp_utc", "event"])
    calendar["timestamp_utc"] = pd.to_datetime(calendar["timestamp_utc"], utc=True)
    calendar = calendar.sort_values(["timestamp_utc", "event"]).reset_index(drop=True)
    return calendar, pd.DataFrame(untimed, columns=["date", "currency", "event", "source"])


def expected_count(per_year: int, year: int, start: date, end: date) -> int:
    """Nominal events in the part of `year` inside the window (pro rata by whole months)."""
    first, last = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
    months = last.month - first.month + 1
    return round(per_year * months / 12)


def coverage_by_year(
    calendar: pd.DataFrame, untimed: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Expected vs found events per category and year; `matched` never exceeds `expected`."""
    rows = []
    for event, category in CATEGORIES.items():
        for year in _years(start, end):
            with_time = int(
                ((calendar["event"] == event) & (calendar["timestamp_utc"].dt.year == year)).sum()
            )
            without_time = int(
                (
                    (untimed["event"] == event) & (pd.to_datetime(untimed["date"]).dt.year == year)
                ).sum()
            )
            expected = expected_count(category.per_year, year, start, end)
            rows.append(
                {
                    "event": event,
                    "year": year,
                    "expected": expected,
                    "found_with_time": with_time,
                    "found_without_time": without_time,
                    "matched": min(with_time, expected),
                }
            )
    return pd.DataFrame(rows)


def summarize_coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Coverage per category and pooled; passes only if every row reaches MIN_COVERAGE."""
    totals = table.groupby("event", sort=False)[["expected", "found_with_time", "matched"]].sum()
    totals.loc["ALL"] = totals.sum()
    totals["coverage"] = totals["matched"] / totals["expected"]
    totals["passes"] = totals["coverage"] >= MIN_COVERAGE
    return totals


# ---------------------------------------------------------------------------------------------
# Market check: does the M1 range peak where the release is expected?
# ---------------------------------------------------------------------------------------------


def load_minute_ranges(m1_dir: Path, pair: str) -> pd.Series:
    """Mid high-low range of every M1 bar of a pair, indexed by UTC bar start."""
    frames = [
        pd.read_parquet(path, columns=["bid_high", "bid_low", "ask_high", "ask_low"])
        for path in sorted(m1_dir.glob(f"{pair}_M1_*.parquet"))
    ]
    if not frames:
        raise FileNotFoundError(f"no M1 parquet for {pair} in {m1_dir}")
    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    return ((bars["ask_high"] + bars["bid_high"]) - (bars["ask_low"] + bars["bid_low"])) / 2


@dataclass(frozen=True)
class Reaction:
    """Where the widest minute around an expected release fell, and how much wider it was."""

    peak: pd.Timestamp
    offset_minutes: int
    burst_ratio: float


def find_reaction(
    ranges: pd.Series,
    expected: pd.Timestamp,
    *,
    before: int = 10,
    after: int = 10,
    baseline: int = 60,
) -> Reaction | None:
    """Widest minute within [expected - before, expected + after], against the earlier hour."""
    window = ranges.loc[
        expected - pd.Timedelta(minutes=before) : expected + pd.Timedelta(minutes=after)
    ]
    quiet = ranges.loc[
        expected - pd.Timedelta(minutes=before + baseline) : expected
        - pd.Timedelta(minutes=before + 1)
    ]
    quiet = quiet[quiet > 0]
    if window.empty or quiet.empty:
        return None
    peak = window.idxmax()
    return Reaction(
        peak=peak,
        offset_minutes=int((peak - expected) / pd.Timedelta(minutes=1)),
        burst_ratio=float(window.max() / quiet.median()),
    )


def _shifted_burst(ranges: pd.Series, expected: pd.Timestamp) -> float:
    """Strongest burst one hour off the expected time; a wrong daylight-saving offset shows here."""
    ratios = [
        reaction.burst_ratio
        for shift in (-60, 60)
        if (reaction := find_reaction(ranges, expected + pd.Timedelta(minutes=shift)))
    ]
    return max(ratios, default=0.0)


def reaction_table(calendar: pd.DataFrame, ranges_by_pair: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per timed event: the expected minute, where the market moved and by how much."""
    rows = []
    for event in calendar.itertuples():
        ranges = ranges_by_pair[PAIR_BY_CURRENCY[event.currency]]
        reaction = find_reaction(ranges, event.timestamp_utc)
        rows.append(
            {
                "timestamp_utc": event.timestamp_utc,
                "event": event.event,
                "peak_utc": reaction.peak if reaction else pd.NaT,
                "offset_minutes": reaction.offset_minutes if reaction else None,
                "burst_ratio": reaction.burst_ratio if reaction else None,
                "shifted_burst_ratio": _shifted_burst(ranges, event.timestamp_utc),
            }
        )
    return pd.DataFrame(rows)


def summarize_reactions(table: pd.DataFrame) -> pd.DataFrame:
    """Per category: events with market data, clear bursts, and where those bursts fall.

    `shifted_hour_stronger` counts events with a clear burst an hour before or after the expected
    time that is stronger than the one at the expected time: a wrong daylight-saving offset would
    fail exactly that test. Press conferences and same-day releases explain the few that remain.
    """
    table = table.assign(with_data=table["burst_ratio"].notna())
    table["clear"] = table["burst_ratio"] >= BURST_RATIO
    table["on_minute"] = table["clear"] & (table["offset_minutes"] == 0)
    table["within_one"] = table["clear"] & (table["offset_minutes"].abs() <= 1)
    table["shifted_hour_stronger"] = table["with_data"] & (
        (table["shifted_burst_ratio"] >= BURST_RATIO)
        & (table["shifted_burst_ratio"] > table["burst_ratio"])
    )
    grouped = table.groupby("event", sort=False)
    return pd.DataFrame(
        {
            "events": grouped.size(),
            "with_m1_data": grouped["with_data"].sum(),
            "clear_burst": grouped["clear"].sum(),
            "on_expected_minute": grouped["on_minute"].sum(),
            "within_one_minute": grouped["within_one"].sum(),
            "shifted_hour_stronger": grouped["shifted_hour_stronger"].sum(),
            "median_burst_ratio": grouped["burst_ratio"].median(),
        }
    )


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------


def _emit(line: str = "") -> None:
    sys.stdout.write(line + "\n")


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=_parse_day, default=WINDOW_START)
    parser.add_argument("--end", type=_parse_day, default=WINDOW_END)
    parser.add_argument("--workdir", type=Path, default=Path("data/raw/calendar"))
    parser.add_argument("--m1-dir", type=Path, default=Path("data/raw/fx_m1_hd"))
    parser.add_argument("--contact", help="address or URL put in the User-Agent (BLS requires one)")
    parser.add_argument(
        "--refresh", action="store_true", help="download again instead of the cache"
    )
    parser.add_argument("--skip-m1", action="store_true", help="do not compare with market data")
    args = parser.parse_args(argv)

    fetcher = Fetcher(args.workdir / "raw", contact=args.contact, refresh=args.refresh)
    try:
        events = collect_all(fetcher, args.start, args.end)
    except FetchError as error:
        _emit(f"fetch failed: {error}")
        return 2
    calendar, untimed = normalize(events, args.start, args.end)
    args.workdir.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(args.workdir / "events.parquet")
    calendar.to_csv(args.workdir / "events.csv", index=False)
    untimed.to_csv(args.workdir / "events_without_time.csv", index=False)

    by_year = coverage_by_year(calendar, untimed, args.start, args.end)
    by_year.to_csv(args.workdir / "coverage_by_year.csv", index=False)
    summary = summarize_coverage(by_year)
    summary.to_csv(args.workdir / "coverage_summary.csv")
    _emit(f"{len(calendar)} events with a published time, {len(untimed)} without\n")
    _emit(summary.to_string(float_format=lambda value: f"{value:.3f}"))

    if not args.skip_m1:
        ranges = {
            pair: load_minute_ranges(args.m1_dir, pair)
            for pair in sorted(set(PAIR_BY_CURRENCY.values()))
        }
        reactions = reaction_table(calendar, ranges)
        reactions.to_csv(args.workdir / "m1_reaction.csv", index=False)
        _emit("\nM1 range peak against the expected minute:")
        _emit(summarize_reactions(reactions).to_string(float_format=lambda value: f"{value:.2f}"))
    return 0 if bool(summary["passes"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
