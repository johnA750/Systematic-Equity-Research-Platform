from html.parser import HTMLParser
from urllib.request import Request, urlopen


SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class _ConstituentTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_cell = False
        self.row = []
        self.cell_text = []
        self.headers = []
        self.records = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "constituents":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.row = []
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_text = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text.append(data)

    def handle_endtag(self, tag):
        if self.in_table and tag in {"th", "td"} and self.in_cell:
            self.row.append("".join(self.cell_text).strip())
            self.in_cell = False
        elif self.in_table and tag == "tr" and self.row:
            if self.row[0] == "Symbol":
                self.headers = self.row
            elif self.headers:
                self.records.append(dict(zip(self.headers, self.row)))
        elif self.in_table and tag == "table":
            self.in_table = False


def normalize_yahoo_ticker(ticker: str) -> str:
    """
    Convert exchange-style tickers to symbols accepted by Yahoo Finance.

    Yahoo Finance uses a hyphen for share classes, for example BRK-B
    instead of BRK.B.
    """
    return str(ticker).strip().upper().replace(".", "-")


def get_sp500_constituents(url: str = SP500_CONSTITUENTS_URL) -> list[dict[str, str]]:
    """
    Return current S&P 500 constituent metadata in Yahoo Finance symbol format.

    The source is intentionally fetched at runtime because this module
    represents the current index membership rather than a historical
    point-in-time universe.
    """
    request = Request(url, headers={"User-Agent": "Trading-Backtester/1.0"})
    with urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8")

    parser = _ConstituentTableParser()
    parser.feed(page)

    if not parser.records:
        raise ValueError("Could not locate the S&P 500 constituent table.")

    constituents = []
    seen = set()
    for record in parser.records:
        symbol = normalize_yahoo_ticker(record["Symbol"])
        if symbol in seen:
            continue
        seen.add(symbol)
        constituents.append(
            {
                "ticker": symbol,
                "security": record.get("Security", ""),
                "sector": record.get("GICS Sector", ""),
                "sub_industry": record.get("GICS Sub-Industry", ""),
            }
        )
    return constituents


def get_sp500_tickers(url: str = SP500_CONSTITUENTS_URL) -> list[str]:
    """Return the current S&P 500 constituent symbols."""
    return [record["ticker"] for record in get_sp500_constituents(url)]


def get_sp500_sector_map(url: str = SP500_CONSTITUENTS_URL) -> dict[str, str]:
    """Return a mapping from current S&P 500 tickers to GICS sectors."""
    return {
        record["ticker"]: record["sector"]
        for record in get_sp500_constituents(url)
        if record["sector"]
    }
