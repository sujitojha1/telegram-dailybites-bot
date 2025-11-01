# Telegram Dailybits Bot

## Background & Objective

Add support for Gmail, Telegram, GDrive (manage your credentials well)
Send message to your Agent on Telegram: "Find latest articles, papers and blogs in hackernews, huggingface paper & AI companies blogs", then put that into a Google Excel Sheet, and then share the link to this sheet with yourself on Gmail, and share the screenshot (or youtube video) on the LMS". Again:
Send a message from Telegram (this message becomes your Agent's query)
Message goes to Agent. Then something should be stored on a Google Sheet. 
Then this sheet link should be sent to you on Gmail.
All of the above is happening via MCP server function calls via the LLM function_call
One of the servers that you add must be an SSE server
