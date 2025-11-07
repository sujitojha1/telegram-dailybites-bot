# Telegram Dailybits Bot

## Background & Objective

This project adds support for Gmail, Telegram, and Google Drive (manage your credentials carefully). The workflow is:

1. Send a message to your agent on Telegram: "Find the latest articles, papers, and blogs in Hacker News, Hugging Face papers, and AI company blogs."
2. The agent stores the results in a Google Sheet.
3. The agent emails the sheet link to you on Gmail.

All of the above happens through MCP server function calls triggered via the LLM `function_call`.
At least one of the servers you add must be an SSE server.


## High Level Plan
- [x] Consolidate existing research, code snippets, and MCP references in the repo; resolve initial embedding/MCP inspection blockers.
- [X] Stand up an `mcp_servers/` package, migrate the current prototype scripts, and define shared base classes/utilities.
- [X] Rebuild the core agent loop with clean config loading (Pydantic), `uv` task scheduling, and structured logging.
- [X] Implement the news discovery workflow (Hacker News, Hugging Face papers, AI company blogs) with reproducible tests.
- [X] Integrate Gmail MCP actions end-to-end, including credential storage, token refresh, and email composition.
- [X] Add a Google Sheets MCP action to persist curated results and expose sheet metadata back to the agent.
- [ ] Design and wire a Telegram bridge that emits updates over SSE, covering webhook/long-poll strategies and failure handling.
- [ ] Establish an async debugging toolkit (pdb, `asyncio` inspectors) and document hot paths for faster triage.
- [ ] Validate the complete flow locally, then choose a hosting strategy for a long-running Telegram listener (e.g., uvicorn worker, serverless).

## References
- [python-mcp Gmail server example](https://github.com/modelcontextprotocol/python-mcp/tree/main/examples/gmail) — ready-to-run FastMCP server exposing list/send Gmail actions (token caching via OAuth2).
- [python-mcp Google Sheets server example](https://github.com/modelcontextprotocol/python-mcp/tree/main/examples/google_sheets) — demonstrates a Sheets MCP server with create/update worksheet tools and shared `google_auth.py` helpers.
- [Gmail API quickstart script](https://github.com/googleworkspace/python-samples/blob/main/gmail/quickstart/quickstart.py) — minimal OAuth flow and message listing code used while wiring the MCP Gmail actions.
- [Sheets API snippets](https://github.com/googleworkspace/python-samples/blob/main/sheets/snippets/sheets_snippets.py) — canonical CRUD operations for Sheets that map directly into the planned MCP tool surface.
- [Starlette server-sent events example](https://github.com/encode/starlette/blob/master/examples/events.py) — production-grade SSE endpoint pattern adaptable for forwarding Telegram updates.
- [python-telegram-bot webhook app](https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/webhookapp.py) — webhook-to-background worker pipeline forming the upstream feed for the SSE bridge.
