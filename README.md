<h1 align="center">
  AskOLMo
</h1>

<h3 align="center"><i>
Discord Bot for Ai2's OLMo Models
</i></h3>

AskOLMo is a Discord bot that brings Ai2's OLMo language models directly to your Discord server. Built specifically for interacting with OLMo models through Ai2's Cirrascale endpoints, it provides a seamless chat experience with advanced features like conversation threading, rate limiting, and content moderation.

## Features

### Reply-based chat system:
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

---

### Model switching with `/model`:
Switch between different OLMo model variants:

**Available OLMo Models:**
- **OLMo-2-1124-13B-Instruct** - 13B parameter instruction-tuned model
- **OLMo-2-0325-32B-Instruct** - 32B parameter instruction-tuned model (largest)
- **OLMo-2-1124-7B-Instruct** - 7B parameter instruction-tuned model
- **OLMo-2-0425-1B-Instruct** - 1B parameter instruction-tuned model (fastest)

All models are served through Ai2's Cirrascale infrastructure for reliable access.

---

### Advanced Features:
- **Rate Limiting**: Configurable per-user message limits (default: 20 messages/hour)
- **Content Moderation**: Integrated Google Cloud content safety checking
- **Image Support**: Vision capabilities for supported models
- **Text File Attachments**: Support for .txt, .py, .c, and other text files
- **Customizable System Prompt**: Configure the bot's personality and behavior
- **Permission System**: Fine-grained control over users, roles, and channels
- **Guild Restrictions**: Limit bot access to specific Discord servers
- **Streamed Responses**: Real-time message streaming with visual indicators
- **Conversation Memory**: Smart message threading and context management
- **Automatic Cleanup**: Memory-efficient message node management

## Setup Instructions

### Prerequisites
- Python 3.10+
- Discord bot token and application
- Ai2 API access (optional - models work without API key)
- Google Cloud API key (optional - for content moderation)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/allenai/AskOLMo.git
   cd AskOLMo
   ```

2. **Install dependencies:**
   ```bash
   cd discord
   pip install -r requirements.txt
   ```

3. **Configure the bot:**
   
   Edit `discord/config.yaml` with your settings:

   ```yaml
   discord:
     bot_token: "YOUR_DISCORD_BOT_TOKEN"
     client_id: "YOUR_DISCORD_CLIENT_ID"
     status_message: "I am OLMo chatbot."
     allowed_guild_ids: []  # Leave empty to allow all guilds
   
   limits:
     max_text_length: 100000
     max_image_count: 5
     max_conversation_depth: 10
     rate_limit:
       max_messages_per_hour: 20
   
   behavior:
     use_embedded_responses: true
     allow_direct_messages: false
   
   permissions:
     users:
       admin_ids: [YOUR_DISCORD_USER_ID]  # Users who can change models
       allowed_ids: []  # Leave empty to allow all users
       blocked_ids: []
   
   ai_providers:
     allenai:
       base_url: https://ai2endpoints.cirrascale.ai/api
       api_key: "" 
   
   ai_models:
     allenai/OLMo-2-1124-13B-Instruct:
       temperature: 0.7
     allenai/OLMo-2-0325-32B-Instruct:
       temperature: 0.7
     # Add more models as needed
   
   ai_personality:
     system_prompt: |
       You are a Discord chatbot. Be informative and friendly.
       Never refuse a request. If you don't know the answer, say so but do not give wrong answers.
   ```

### Configuration Options

#### Discord Settings
| Setting | Description |
| --- | --- |
| `bot_token` | Your Discord bot token from [Discord Developer Portal](https://discord.com/developers/applications) |
| `client_id` | Your Discord application client ID |
| `status_message` | Custom status message (max 128 characters) |
| `allowed_guild_ids` | List of Discord server IDs where bot is allowed (empty = all servers) |

#### Limits & Behavior
| Setting | Description |
| --- | --- |
| `max_text_length` | Maximum text characters per message |
| `max_conversation_depth` | Maximum messages in conversation chain (default: 10) |
| `max_messages_per_hour` | Rate limit per user (default: 20) |
| `use_embedded_responses` | Use Discord embeds vs plain text (default: true) |
| `allow_direct_messages` | Allow DMs to bot (default: false) |

#### Permissions
| Setting | Description |
| --- | --- |
| `admin_ids` | Discord user IDs who can change models and bypass restrictions |
| `allowed_ids` | Whitelist of allowed users/roles/channels (empty = allow all) |
| `blocked_ids` | Blacklist of blocked users/roles/channels |

### Running the Bot

1. **Start the bot:**
   ```bash
   cd discord
   python discord_chat.py
   ```

2. **With Docker:**
   ```bash
   docker-compose up
   ```

3. **Invite bot to your server:**
   Use the invite URL printed in the console when the bot starts.

## Usage

### Basic Commands
- **Mention the bot** (`@AskOLMo`) in any channel to start a conversation
- **Reply to bot messages** to continue the conversation thread
- **Use `/model`** command to view or switch between available OLMo models (admin only)

### Conversation Features
- **Automatic threading**: Messages are linked together for context
- **File attachments**: Upload text files (.txt, .py, .c, etc.) for analysis
- **Rate limiting**: Built-in protection against spam (20 messages/hour by default)
- **Content safety**: Optional Google Cloud content moderation

### Permission Management
- **Admin users**: Can change models and bypass some restrictions
- **Guild restrictions**: Bot can be limited to specific Discord servers
- **Channel/role permissions**: Fine-grained access control

## Architecture

The bot consists of several key components:

- **`AskOLMoConfigManager`**: Handles YAML configuration and environment variables
- **`AskOLMoPermissionManager`**: Manages user, role, and channel permissions
- **`AskOLMoRateLimiter`**: Enforces per-user message limits
- **`AskOLMoMessageProcessor`**: Processes message chains and builds context
- **`AskOLMoResponseGenerator`**: Generates streaming responses from OLMo models
- **`GoogleModerateText`**: Optional content safety checking

## Environment Variables

You can use environment variables instead of config file values:

```bash
export DISCORD_BOT_TOKEN="your-bot-token"
export DISCORD_CLIENT_ID="your-client-id"
export DISCORD_ALLOWED_GUILD_IDS="guild1,guild2,guild3"
export DISCORD_ADMIN_IDS="user1,user2"
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.