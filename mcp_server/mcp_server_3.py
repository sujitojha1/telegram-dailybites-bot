from mcp.server.fastmcp import FastMCP, Context
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import urllib.parse
import sys
import traceback
import asyncio
from datetime import datetime, timedelta
import time
import re
import os
import base64
import html
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, AnyHttpUrl, ValidationError, field_validator

load_dotenv()

GOOGLE_CREDS_PATH = "/Users/payalchakraborty/Dev/google/client_creds.json"
GMAIL_TOKEN_PATH = "/Users/payalchakraborty/Dev/google/app_tokens.json"
SHEETS_TOKEN_PATH = "/Users/payalchakraborty/Dev/google/sheets_tokens.json"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DAILYBITES_SPREADSHEET_ID = os.getenv("DAILYBITES_SPREADSHEET_ID", "").strip()
DAILYBITES_SHEET_RANGE = os.getenv("DAILYBITES_SHEET_RANGE", "Sheet1!A:D").strip()


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


@dataclass
class EmailContent:
    subject: str
    message: str


@dataclass
class SheetRow:
    date: str
    link1: str
    link2: str
    link3: str


class EmailRequest(BaseModel):
    subject: str = Field(..., min_length=3, max_length=200, description="Email subject line")
    message: str = Field(..., min_length=1, description="Email body text or HTML")

    @field_validator("subject", "message", mode="before")
    @classmethod
    def _strip_whitespace(cls, value: Any) -> str:
        if value is None:
            raise ValueError("Value cannot be null")
        value = str(value).strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value

    def to_dataclass(self) -> EmailContent:
        return EmailContent(subject=self.subject, message=self.message)


class SheetAppendRequest(BaseModel):
    link1: AnyHttpUrl
    link2: AnyHttpUrl
    link3: AnyHttpUrl
    date_str: Optional[str] = Field(default=None, description="ISO date string (YYYY-MM-DD)")

    @field_validator("date_str")
    @classmethod
    def _validate_date(cls, value: Optional[str]) -> Optional[str]:
        if value:
            datetime.strptime(value, "%Y-%m-%d")
        return value

    def to_dataclass(self) -> SheetRow:
        resolved_date = self.date_str or datetime.now().strftime("%Y-%m-%d")
        return SheetRow(
            date=resolved_date,
            link1=str(self.link1),
            link2=str(self.link2),
            link3=str(self.link3),
        )


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class DuckDuckGoSearcher:
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter()

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results in a natural language style that's easier for LLMs to process"""
        if not results:
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes."

        header = "DAILYBITES SEARCH DIGEST"
        sub_header = f"Top {len(results)} curated links"

        output = [
            header,
            "=" * len(header),
            sub_header,
            "-" * len(sub_header),
            "",
            "Detailed Highlights",
            "--------------------",
        ]

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   Link   : {result.link}")
            if result.snippet:
                output.append(f"   Summary: {result.snippet}")
            output.append("")

        output.extend(
            [
                "Quick Link List",
                "---------------",
            ]
        )
        for result in results:
            output.append(f"[{result.position}] {result.link}")

        return "\n".join(output).strip()

    async def search(
        self, query: str, ctx: Context, max_results: int = 10
    ) -> List[SearchResult]:
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Create form data for POST request
            data = {
                "q": query,
                "b": "",
                "kl": "",
            }

            await ctx.info(f"Searching DuckDuckGo for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL, data=data, headers=self.HEADERS, timeout=30.0
                )
                response.raise_for_status()

            # Parse HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            if not soup:
                await ctx.error("Failed to parse HTML response")
                return []

            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title")
                if not title_elem:
                    continue

                link_elem = title_elem.find("a")
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")

                # Skip ad results
                if "y.js" in link:
                    continue

                # Clean up DuckDuckGo redirect URLs
                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])

                snippet_elem = result.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(
                    SearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        position=len(results) + 1,
                    )
                )

                if len(results) >= max_results:
                    break

            await ctx.info(f"Successfully found {len(results)} results")
            return results

        except httpx.TimeoutException:
            await ctx.error("Search request timed out")
            return []
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred: {str(e)}")
            return []
        except Exception as e:
            await ctx.error(f"Unexpected error during search: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            return []


class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: Context) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            await ctx.info(f"Fetching content from: {url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"

            await ctx.info(
                f"Successfully fetched and parsed content ({len(text)} characters)"
            )
            return text

        except httpx.TimeoutException:
            await ctx.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await ctx.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"


def strip_html_tags(text: str) -> str:
    """Strip HTML tags for email-safe plain text output."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator="\n")
    cleaned_text = re.sub(r"\n{2,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def is_probably_html(text: str) -> bool:
    if not text:
        return False
    soup = BeautifulSoup(text, "html.parser")
    return bool(soup.find()) or bool(re.search(r"<[^>]+>", text))


def build_html_email_body(message: str, fallback_text: str) -> str:
    """Convert the message into an HTML email body with monospace formatting fallback."""
    base_style = "font-family: 'Segoe UI', Arial, sans-serif; font-size: 15px; color: #111; line-height: 1.5;"
    if is_probably_html(message):
        body_content = message
    else:
        safe_text = html.escape(message or fallback_text or "")
        body_content = (
            "<pre style=\"font-family: 'SFMono-Regular', Consolas, monospace; font-size: 14px; "
            "background-color: #f7f7f9; padding: 16px; border-radius: 8px; line-height: 1.4;\">"
            f"{safe_text}"
            "</pre>"
        )
    return f"<html><body style=\"{base_style}\">{body_content}</body></html>"


def load_google_credentials(scopes: List[str], token_path: str) -> Credentials:
    """Load or refresh OAuth credentials for the requested scopes."""
    creds: Optional[Credentials] = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDS_PATH, scopes)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())

    return creds


# Initialize FastMCP server
mcp = FastMCP("ddg-search")
searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()


@mcp.tool()
async def search(query: str, ctx: Context, max_results: int = 10) -> str:
    """
    Search DuckDuckGo and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: MCP context for logging
    """
    try:
        results = await searcher.search(query, ctx, max_results)
        return searcher.format_results_for_llm(results)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while searching: {str(e)}"


@mcp.tool()
async def fetch_content(url: str, ctx: Context) -> str:
    """
    Fetch and parse content from a webpage URL.

    Args:
        url: The webpage URL to fetch content from
        ctx: MCP context for logging
    """
    return await fetcher.fetch_and_parse(url, ctx)

@mcp.tool()
async def send_email(subject: str, message: str) -> dict:
    """
    Send an email to self using Gmail API.

    Args:
        subject: subject of email
        message: body of email
    """

    try:
        email_payload = EmailRequest(subject=subject, message=message).to_dataclass()
        creds = load_google_credentials(GMAIL_SCOPES, GMAIL_TOKEN_PATH)
        service = build('gmail', 'v1', credentials=creds)

        user_profile = service.users().getProfile(userId='me').execute()
        user_email = user_profile.get('emailAddress', 'me')

        clean_message = strip_html_tags(email_payload.message)
        if not clean_message:
            clean_message = email_payload.message.strip()

        html_message = build_html_email_body(email_payload.message, clean_message)

        message_obj = EmailMessage()
        message_obj.set_content(clean_message, subtype="plain", charset="utf-8")
        message_obj.add_alternative(html_message, subtype="html", charset="utf-8")
        message_obj['To'] = user_email
        message_obj['From'] = user_email
        message_obj['Subject'] = email_payload.subject

        encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
        create_message = {'raw': encoded_message}

        send_message = await asyncio.to_thread(
            service.users().messages().send(userId="me", body=create_message).execute
        )

        return {"status": "success", "message_id": send_message['id']}

    except ValidationError as error:
        return {"status": "error", "error_message": error.errors()}
    except HttpError as error:
        return {"status": "error", "error_message": str(error)}


@mcp.tool()
async def append_google_sheets(
    link1: str,
    link2: str,
    link3: str,
    date_str: Optional[str] = None,
) -> dict:
    """
    Append a row with date + three links to a Google Sheet stored in Drive.

    Args:
        link1/link2/link3: URLs to store.
        date_str: Optional ISO date string. Defaults to today's date if omitted.
    """

    try:
        if not DAILYBITES_SPREADSHEET_ID:
            raise ValueError("DAILYBITES_SPREADSHEET_ID is not configured in the environment.")

        sheet_request = SheetAppendRequest(
            link1=link1,
            link2=link2,
            link3=link3,
            date_str=date_str,
        ).to_dataclass()

        spreadsheet_id = DAILYBITES_SPREADSHEET_ID
        sheet_range = DAILYBITES_SHEET_RANGE or "Sheet1!A:D"
        creds = load_google_credentials(SHEETS_SCOPES, SHEETS_TOKEN_PATH)
        service = build('sheets', 'v4', credentials=creds)

        payload = [[sheet_request.date, sheet_request.link1, sheet_request.link2, sheet_request.link3]]

        request = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=sheet_range,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": payload},
        )

        response = await asyncio.to_thread(request.execute)
        updates = response.get("updates", {})

        return {
            "status": "success",
            "updatedRange": updates.get("updatedRange"),
            "updatedRows": updates.get("updatedRows", 0),
        }

    except ValidationError as error:
        return {"status": "error", "error_message": error.errors()}
    except (HttpError, ValueError) as error:
        return {"status": "error", "error_message": str(error)}



if __name__ == "__main__":
    print("mcp_server_3.py starting")
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
            mcp.run()  # Run without transport for dev server
    else:
        import threading
        server_thread = threading.Thread(target=lambda: mcp.run(transport="stdio"))
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)

        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
