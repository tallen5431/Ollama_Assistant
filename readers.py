#!/usr/bin/env python3
"""How each kind of document is turned into text the model can read.

A reader says which documents it handles and how to read them; ``web.fetch``
downloads the bytes and hands them to the first reader that accepts them.
Adding a kind of document — a spreadsheet, an e-book — is adding a reader
here, without touching the fetching, the address guard or the pipeline that
uses the text. The HTML, plain-text and JSON readers register themselves from
``web.py``, where the HTML extractor lives.

Two readers live here because they need things the rest of the app does not:

* **PDF**, with ``pypdf``. Manuals, data sheets, papers and council minutes are
  PDFs far more often than web pages, and they were refused outright as
  "application/pdf, which has no text to read".
* **A browser**, with Playwright, for a page that is empty until its
  JavaScript runs. Off unless WEB_RENDER=1; see ``render``.
"""

from __future__ import annotations

import io
import logging
from urllib.parse import urljoin
from typing import Any, Callable, Dict, List, NamedTuple, Optional

logger = logging.getLogger("ollama-chat")


class ReadError(Exception):
    """A document that was fetched but could not be read. Said to the user."""


class Reader(NamedTuple):
    """One kind of document.

    ``accepts(content_type, url, head)`` says whether this reader handles it —
    ``head`` is the first bytes, because a server calling a PDF
    "application/octet-stream" is common. ``read(raw, content_type, url)``
    returns ``{"title", "text", "links"}`` or raises ReadError. ``max_bytes``
    is how much may be downloaded for it.
    """
    name: str
    accepts: Callable[[str, str, bytes], bool]
    read: Callable[[bytes, str, str], Dict[str, Any]]
    max_bytes: Callable[[], int]


_READERS: List[Reader] = []


def register(reader: Reader, first: bool = False) -> None:
    """Add a reader. ``first`` puts it ahead of the ones already there."""
    global _READERS
    others = [r for r in _READERS if r.name != reader.name]
    _READERS = [reader] + others if first else others + [reader]


def reader_for(content_type: str, url: str, head: bytes = b"") -> Optional[Reader]:
    """The first reader that accepts this document, or None."""
    for reader in _READERS:
        try:
            if reader.accepts(content_type, url, head):
                return reader
        except Exception:  # noqa: BLE001 - a broken reader must not take the others down
            logger.exception("Reader %s failed to decide on %s", reader.name, url)
    return None


def names() -> List[str]:
    return [r.name for r in _READERS]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# Enough for a long manual. Text past this is cut by the page budget anyway,
# and a 900-page PDF is slow to walk for no gain.
_PDF_MAX_PAGES = 150
_PDF_LINKS_MAX = 120


def _is_pdf(content_type: str, url: str, head: bytes) -> bool:
    if head.startswith(b"%PDF-"):
        return True
    if content_type == "application/pdf":
        return True
    return content_type in ("", "application/octet-stream", "binary/octet-stream") \
        and url.lower().split("?", 1)[0].endswith(".pdf")


def read_pdf(raw: bytes, content_type: str, url: str,
             max_chars: int = 60000) -> Dict[str, Any]:
    """The text of a PDF, page by page, with its page numbers kept.

    Page numbers stay in the text ("[page 12]") because a figure from a
    manual is only checkable if you can find it again.
    """
    try:
        from pypdf import PdfReader
    except Exception as exc:  # noqa: BLE001 - missing or broken install
        raise ReadError("PDFs need pypdf on this machine — "
                        "pip install -r requirements.txt") from exc
    try:
        pdf = PdfReader(io.BytesIO(raw))
        if pdf.is_encrypted:
            # Many PDFs are "encrypted" with an empty password just to set
            # permissions; those open fine. A real password is a refusal.
            try:
                if not pdf.decrypt(""):
                    raise ReadError(f"{url} is a password-protected PDF")
            except ReadError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ReadError(f"{url} is a protected PDF that could not be opened") from exc
        title = ""
        try:
            title = str((pdf.metadata or {}).get("/Title") or "").strip()
        except Exception:  # noqa: BLE001 - metadata is optional and often malformed
            title = ""
        parts: List[str] = []
        links: List[Dict[str, str]] = []
        total = 0
        count = len(pdf.pages)
        for number, page in enumerate(pdf.pages[:_PDF_MAX_PAGES], start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:  # noqa: BLE001 - one bad page is not the document
                text = ""
            if text:
                parts.append(f"[page {number}]\n{text}")
                total += len(text)
            links.extend(_pdf_links(page))
            if total >= max_chars:
                break
    except ReadError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises many things on bad files
        raise ReadError(f"{url} is a PDF that could not be read ({exc.__class__.__name__})") from exc
    if not parts:
        # The whole document is pictures of pages: a scan. Said plainly, since
        # "no readable text" otherwise sounds like a fetching problem.
        raise ReadError(f"{url} is a scanned PDF with no text layer to read")
    if count > _PDF_MAX_PAGES and total < max_chars:
        parts.append(f"[… the PDF has {count} pages; the first {_PDF_MAX_PAGES} were read]")
    return {"title": title or url, "text": "\n\n".join(parts),
            "links": links[:_PDF_LINKS_MAX]}


def _pdf_links(page: Any) -> List[Dict[str, str]]:
    """Web links on one PDF page, for the "what else is here" list."""
    out: List[Dict[str, str]] = []
    try:
        annotations = page.get("/Annots") or []
        for ref in annotations:
            note = ref.get_object()
            action = note.get("/A") or {}
            uri = str(action.get("/URI") or "")
            if uri.startswith(("http://", "https://")):
                out.append({"url": uri, "text": ""})
    except Exception:  # noqa: BLE001 - annotations are optional and often malformed
        return out
    return out


def pdf_reader(max_bytes: Callable[[], int],
               max_chars: Callable[[], int]) -> Reader:
    return Reader("pdf", _is_pdf,
                  lambda raw, ctype, url: read_pdf(raw, ctype, url, max_chars()),
                  max_bytes)


# ---------------------------------------------------------------------------
# A browser, for pages that are empty without JavaScript
# ---------------------------------------------------------------------------

# Requests a page makes that cost time and carry no text.
_SKIP_RESOURCES = frozenset({"image", "media", "font", "stylesheet", "manifest",
                             "texttrack", "eventsource", "websocket", "other"})
_RENDER_REDIRECTS = 5
_RENDER_ARGS = [
    # WebRTC can reach the local network without going through any request a
    # route handler sees; turn its address discovery off.
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--disable-features=WebRtcHideLocalIpsWithMdns",
]


def render_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def render(url: str, guard: Callable[[str], str], timeout: float,
           user_agent: str) -> Dict[str, str]:
    """Open ``url`` in headless Chromium and return ``{"url", "html"}``.

    Every request the page makes — the page itself, its scripts, its API calls,
    and each hop of every redirect — goes through ``guard``, the same check
    that stops an ordinary fetch reaching a private address. The browser is
    not trusted to follow redirects on its own: each request is made with
    redirects off and the redirect handed back to the browser, so the next hop
    comes back through the guard too. Images, fonts, media and websockets are
    not loaded at all.

    What the guard cannot cover is DNS rebinding between its check and the
    browser's own lookup — the plain fetch path pins the address it checked,
    and a browser cannot be made to. That gap is why this is opt-in.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        raise ReadError("Opening pages in a browser needs Playwright — "
                        "pip install playwright && playwright install chromium") from exc

    blocked: List[str] = []

    def route(handled: Any) -> None:
        request = handled.request
        if request.resource_type in _SKIP_RESOURCES:
            handled.abort()
            return
        # Redirects are followed here, one checked hop at a time. Handing a
        # redirect back to the browser does not work: it follows it natively,
        # and the request it makes for the new address never comes back
        # through this handler — so a public page redirecting to the router's
        # admin page would have been loaded unchecked.
        url, method = request.url, request.method
        response = None
        for _ in range(_RENDER_REDIRECTS):
            try:
                guard(url)
            except Exception as exc:  # noqa: BLE001 - the guard's own refusal
                blocked.append(f"{url}: {exc}")
                handled.abort()
                return
            try:
                response = handled.fetch(url=url, method=method, max_redirects=0,
                                         timeout=timeout * 1000)
            except Exception:  # noqa: BLE001 - one failed script is not the page
                handled.abort()
                return
            location = response.headers.get("location") or ""
            if response.status not in (301, 302, 303, 307, 308) or not location:
                break
            url = urljoin(url, location)
            if response.status in (301, 302, 303):
                method = "GET"
        else:
            handled.abort()
            return
        handled.fulfill(response=response)

    ms = int(timeout * 1000)
    with sync_playwright() as driver:
        browser = driver.chromium.launch(args=_RENDER_ARGS)
        try:
            context = browser.new_context(user_agent=user_agent,
                                          service_workers="block",
                                          accept_downloads=False)
            context.route("**/*", route)
            # WebSockets never pass through route(): closed outright, so a page
            # cannot open one to an address the guard would have refused.
            if hasattr(context, "route_web_socket"):
                context.route_web_socket("**/*", lambda socket: socket.close())
            page = context.new_page()
            page.set_default_timeout(ms)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=ms)
            except Exception as exc:  # noqa: BLE001
                if blocked:
                    raise ReadError(f"Refused while opening {url}: {blocked[0]}") from exc
                raise ReadError(f"Could not open {url} in a browser: {exc}") from exc
            try:
                # Most single-page apps fill in within a second or two of the
                # document loading. Waiting for the network to go quiet is the
                # best general signal, bounded so a page that polls forever
                # does not hold the turn.
                page.wait_for_load_state("networkidle", timeout=min(ms, 5000))
            except Exception:  # noqa: BLE001 - quiet enough is fine
                pass
            final = page.url
            guard(final)
            html = page.content()
        finally:
            browser.close()
    return {"url": final, "html": html}
