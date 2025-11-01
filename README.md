# Telegram Dailybits Bot

## Background & Objective

This project adds support for Gmail, Telegram, and Google Drive (manage your credentials carefully). The workflow is:

1. Send a message to your agent on Telegram: "Find the latest articles, papers, and blogs in Hacker News, Hugging Face papers, and AI company blogs."
2. The agent stores the results in a Google Sheet.
3. The agent emails the sheet link to you on Gmail.
4. The agent shares the screenshot (or YouTube video) on the LMS.

All of the above happens through MCP server function calls triggered via the LLM `function_call`.
At least one of the servers you add must be an SSE server.


## High Level Plan
- [ ] Gather relevant code and references into a folder and review them.

## References
- [Model Context Protocol Overview](https://modelcontextprotocol.io/introduction) — article explaining MCP essentials for orchestrating tool-augmented agents. (Tentative rank 1)
- [modelcontextprotocol/python-mcp](https://github.com/modelcontextprotocol/python-mcp) — GitHub repository with reference MCP server implementations ready to extend for Gmail and Sheets. (Tentative rank 2)
- [Google Developers: Gmail API Python Quickstart](https://developers.google.com/gmail/api/quickstart/python) — article covering the auth and message operations you will wrap as MCP tools. (Tentative rank 3)
- [Google Developers: Sheets API Python Quickstart](https://developers.google.com/sheets/api/quickstart/python) — article showing how to persist agent output in Sheets through an MCP connector. (Tentative rank 4)
- [googleworkspace/python-samples](https://github.com/googleworkspace/python-samples) — GitHub repository with end-to-end Gmail and Sheets samples to adapt into MCP actions. (Tentative rank 5)
- [Telegram Bot API Documentation](https://core.telegram.org/bots/api) — article detailing webhook and long-poll patterns to pair with an SSE bridge for Telegram updates. (Tentative rank 6)
- [MDN Web Docs: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) — article explaining SSE fundamentals for streaming Telegram bot events. (Tentative rank 7)
- [EventSource/eventsource](https://github.com/EventSource/eventsource) — GitHub repository demonstrating SSE client/server interoperability for bot delivery pipelines. (Tentative rank 8)
