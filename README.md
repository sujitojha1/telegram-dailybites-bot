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

## Referernces 

