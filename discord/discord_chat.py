"""
Discord chatbot for Ai2 models
"""

import asyncio
import logging
import os
from base64 import b64encode
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

import httpx
import yaml
from google_moderation_text import GoogleModerateText
from openai import AsyncOpenAI
from src.message.SafetyChecker import SafetyCheckRequest

import discord
from discord.app_commands import Choice
from discord.ext import commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = "allenai"
PROVIDERS_SUPPORTING_USERNAMES = "allenai"

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " 💭"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


@dataclass
class AskOLMoMsgNode:
    """
    Node representing a message in the conversation chain.
    """

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class AskOLMoConfigManager:
    """
    This class manages the configuration for the Discord bot.
    """

    def __init__(self, filename: str = "config.yaml"):
        self.filename = filename
        self._raw_config = self._load_config()
        self._config = self._transform_config(self._raw_config)

    def _load_config(self) -> dict[str, Any]:
        with open(self.filename, encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _transform_config(self, raw_config: dict[str, Any]) -> dict[str, Any]:

        bot_token = raw_config.get("discord", {}).get("bot_token", "")
        if not bot_token:
            bot_token = os.getenv("DISCORD_BOT_TOKEN", "")

        client_id = raw_config.get("discord", {}).get("client_id", "")
        if not client_id:
            client_id = os.getenv("DISCORD_CLIENT_ID", "")

        allowed_guild_ids = raw_config.get("discord", {}).get("allowed_guild_ids", [])
        if not allowed_guild_ids:
            env_guild_ids = os.getenv("DISCORD_ALLOWED_GUILD_IDS", "")
            if env_guild_ids:
                try:
                    allowed_guild_ids = [
                        int(id.strip()) for id in env_guild_ids.split(",") if id.strip()
                    ]
                except ValueError:
                    allowed_guild_ids = []

        return {
            "bot_token": bot_token,
            "client_id": client_id,
            "status_message": raw_config.get("discord", {}).get("status_message", ""),
            "allowed_guild_ids": allowed_guild_ids,
            "models": raw_config.get("ai_models", {}),
            "providers": raw_config.get("ai_providers", {}),
            "max_text": raw_config.get("limits", {}).get("max_text_length", 100000),
            "max_images": raw_config.get("limits", {}).get("max_image_count", 5),
            "max_messages": raw_config.get("limits", {}).get(
                "max_conversation_depth", 25
            ),
            "rate_limit": raw_config.get("limits", {}).get("rate_limit", {}),
            "use_plain_responses": not raw_config.get("behavior", {}).get(
                "use_embedded_responses", True
            ),
            "allow_dms": raw_config.get("behavior", {}).get(
                "allow_direct_messages", False
            ),
            "system_prompt": raw_config.get("ai_personality", {}).get(
                "system_prompt", ""
            ),
            "permissions": self._process_permissions(raw_config.get("permissions", {})),
        }

    def _process_permissions(self, permissions: dict[str, Any]) -> dict[str, Any]:
        processed_permissions = permissions.copy()
        admin_ids = permissions.get("users", {}).get("admin_ids", [])
        if not admin_ids:
            env_admin_ids = os.getenv("DISCORD_ADMIN_IDS", "")
            if env_admin_ids:
                try:
                    admin_ids = [
                        int(id.strip()) for id in env_admin_ids.split(",") if id.strip()
                    ]
                except ValueError:
                    admin_ids = []

        if "users" not in processed_permissions:
            processed_permissions["users"] = {}
        processed_permissions["users"]["admin_ids"] = admin_ids

        return processed_permissions

    async def reload_config(self) -> dict[str, Any]:
        self._raw_config = await asyncio.to_thread(self._load_config)
        self._config = self._transform_config(self._raw_config)
        return self._config

    @property
    def config(self) -> dict[str, Any]:
        return self._config


class AskOLMoRateLimiter:
    """
    Rate limiter to enforce message limits per user per hour.
    """
    
    def __init__(self, config: dict[str, Any]):
        self.max_messages_per_hour = config.get("rate_limit", {}).get("max_messages_per_hour", 10)
        self.user_message_history = defaultdict(list)
    
    def is_rate_limited(self, user_id: int) -> bool:
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        self.user_message_history[user_id] = [
            timestamp for timestamp in self.user_message_history[user_id]
            if timestamp > one_hour_ago
        ]
        
        return len(self.user_message_history[user_id]) >= self.max_messages_per_hour
    
    def add_message(self, user_id: int) -> None:
        self.user_message_history[user_id].append(datetime.now())
    
    def get_remaining_messages(self, user_id: int) -> int:
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        self.user_message_history[user_id] = [
            timestamp for timestamp in self.user_message_history[user_id]
            if timestamp > one_hour_ago
        ]
        return max(0, self.max_messages_per_hour - len(self.user_message_history[user_id]))
    
    def cleanup_old_data(self) -> None:
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        users_to_remove = []
        for user_id, timestamps in self.user_message_history.items():
            filtered_timestamps = [ts for ts in timestamps if ts > one_hour_ago]
            if not filtered_timestamps:
                users_to_remove.append(user_id)
            else:
                self.user_message_history[user_id] = filtered_timestamps
        
        for user_id in users_to_remove:
            del self.user_message_history[user_id]


class AskOLMoPermissionManager:
    """
    Manages user and channel permissions for the Discord bot.
    """

    def __init__(self, config: dict[str, Any]):
        self.permissions = config["permissions"]

    def is_admin(self, user_id: int) -> bool:
        return user_id in self.permissions["users"]["admin_ids"]

    def check_user_permissions(
        self, user: discord.User, role_ids: set[int], is_dm: bool
    ) -> bool:
        user_is_admin = self.is_admin(user.id)
        allowed_user_ids = self.permissions["users"]["allowed_ids"]
        blocked_user_ids = self.permissions["users"]["blocked_ids"]
        allowed_role_ids = self.permissions["roles"]["allowed_ids"]
        blocked_role_ids = self.permissions["roles"]["blocked_ids"]

        allow_all_users = (
            not allowed_user_ids
            if is_dm
            else not allowed_user_ids and not allowed_role_ids
        )
        is_good_user = (
            user_is_admin
            or allow_all_users
            or user.id in allowed_user_ids
            or any(id in allowed_role_ids for id in role_ids)
        )
        is_bad_user = (
            not is_good_user
            or user.id in blocked_user_ids
            or any(id in blocked_role_ids for id in role_ids)
        )

        return not is_bad_user

    def check_channel_permissions(
        self,
        channel: discord.abc.Messageable,
        channel_ids: set[int],
        is_dm: bool,
        user_id: int,
        allow_dms: bool,
    ) -> bool:
        user_is_admin = self.is_admin(user_id)
        allowed_channel_ids = self.permissions["channels"]["allowed_ids"]
        blocked_channel_ids = self.permissions["channels"]["blocked_ids"]

        allow_all_channels = not allowed_channel_ids
        is_good_channel = (
            user_is_admin or allow_dms
            if is_dm
            else allow_all_channels
            or any(id in allowed_channel_ids for id in channel_ids)
        )
        is_bad_channel = not is_good_channel or any(
            id in blocked_channel_ids for id in channel_ids
        )

        return not is_bad_channel


class AskOLMoResponseGenerator:
    """
    Generates responses for Discord messages using Ai2 models through Cirrascale.
    """

    def __init__(self, config: dict[str, Any], message_processor: "AskOLMoMessageProcessor"):
        self.config = config
        self.httpx_client = httpx.AsyncClient()
        self.last_task_time = 0
        self.message_processor = message_processor

    async def create_openai_client(self, provider: str) -> AsyncOpenAI:
        base_url = self.config["providers"][provider]["base_url"]
        api_key = self.config["providers"][provider].get("api_key", "")

        if not api_key:
            env_var_name = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(env_var_name, "sk-no-key-required")

        return AsyncOpenAI(base_url=base_url, api_key=api_key)

    def supports_vision(self, model: str) -> bool:
        return any(x in model.lower() for x in VISION_MODEL_TAGS)

    def supports_usernames(self, provider_slash_model: str) -> bool:
        return any(
            x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES
        )

    async def generate_response(
        self,
        messages: list[dict],
        model: str,
        model_parameters: dict,
        new_msg: discord.Message,
    ) -> tuple[list[discord.Message], list[str]]:
        provider, model_name = model.split("/", 1)
        cirrascale_client = await self.create_openai_client(provider)

        curr_content = finish_reason = edit_task = None
        response_msgs = []
        response_contents = []

        embed = discord.Embed()
        use_plain_responses = self.config["use_plain_responses"]
        max_message_length = (
            2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))
        )

        try:
            async with new_msg.channel.typing():
                async for curr_chunk in await cirrascale_client.chat.completions.create(
                    model=model_name,
                    messages=messages[::-1],  # type: ignore
                    stream=True,
                    extra_body=model_parameters,  # type: ignore
                ):
                    if finish_reason is not None:
                        break

                    if not (
                        choice := curr_chunk.choices[0] if curr_chunk.choices else None
                    ):
                        continue

                    finish_reason = choice.finish_reason

                    prev_content = curr_content or ""
                    curr_content = choice.delta.content or ""

                    new_content = (
                        prev_content
                        if finish_reason is None
                        else (prev_content + curr_content)
                    )

                    if response_contents == [] and new_content == "":
                        continue

                    if (
                        start_next_msg := response_contents == []
                        or len(response_contents[-1] + new_content) > max_message_length
                    ):
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        ready_to_edit = (
                            edit_task is None or edit_task.done()
                        ) and datetime.now().timestamp() - self.last_task_time >= EDIT_DELAY_SECONDS
                        msg_split_incoming = (
                            finish_reason is None
                            and len(response_contents[-1] + curr_content)
                            > max_message_length
                        )
                        is_final_edit = finish_reason is not None or msg_split_incoming
                        is_good_finish = (
                            finish_reason is not None
                            and finish_reason.lower() in ("stop", "end_turn")
                        )

                        if start_next_msg or ready_to_edit or is_final_edit:
                            if edit_task is not None:
                                await edit_task

                            embed.description = (
                                response_contents[-1]
                                if is_final_edit
                                else (response_contents[-1] + STREAMING_INDICATOR)
                            )
                            embed.color = (
                                EMBED_COLOR_COMPLETE
                                if msg_split_incoming or is_good_finish
                                else EMBED_COLOR_INCOMPLETE
                            )

                            if start_next_msg:
                                reply_to_msg = (
                                    new_msg
                                    if response_msgs == []
                                    else response_msgs[-1]
                                )
                                response_msg = await reply_to_msg.reply(
                                    embed=embed, silent=True
                                )
                                response_msgs.append(response_msg)

                                node = AskOLMoMsgNode(parent_msg=new_msg)
                                self.message_processor.msg_nodes[response_msg.id] = node
                                await node.lock.acquire()
                            else:
                                edit_task = asyncio.create_task(
                                    response_msgs[-1].edit(embed=embed)
                                )

                            self.last_task_time = datetime.now().timestamp()

                if use_plain_responses:
                    for content in response_contents:
                        reply_to_msg = (
                            new_msg if response_msgs == [] else response_msgs[-1]
                        )
                        response_msg = await reply_to_msg.reply(
                            content=content, suppress_embeds=True
                        )
                        response_msgs.append(response_msg)

                        node = AskOLMoMsgNode(parent_msg=new_msg)
                        self.message_processor.msg_nodes[response_msg.id] = node
                        await node.lock.acquire()

        except Exception:
            logging.exception("Error while generating response")

        return response_msgs, response_contents


class AskOLMoMessageProcessor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.msg_nodes = {}
        self.httpx_client = httpx.AsyncClient()

    async def process_message_chain(
        self, new_msg: discord.Message, curr_model: str, bot_user: discord.User
    ) -> tuple[list[dict], set[str]]:
        provider_slash_model = curr_model
        provider, model = provider_slash_model.split("/", 1)

        accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
        accept_usernames = any(
            x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES
        )

        max_text = self.config["max_text"]
        max_images = self.config["max_images"] if accept_images else 0
        max_messages = self.config["max_messages"]

        messages = []
        user_warnings = set()
        curr_msg = new_msg

        while curr_msg is not None and len(messages) < max_messages:
            curr_node = self.msg_nodes.setdefault(curr_msg.id, AskOLMoMsgNode())

            async with curr_node.lock:
                if curr_node.text is None:
                    await self._process_message_content(curr_msg, curr_node, bot_user)  # type: ignore

                content = self._build_message_content(curr_node, max_text, max_images)

                if content != "":
                    message = dict(content=content, role=curr_node.role)
                    if accept_usernames and curr_node.user_id is not None:
                        message["name"] = str(curr_node.user_id)

                    messages.append(message)

                self._collect_warnings(
                    curr_node,
                    max_text,
                    max_images,
                    len(messages),
                    max_messages,
                    user_warnings,
                )
                curr_msg = curr_node.parent_msg

        if system_prompt := self.config["system_prompt"]:
            messages.append(self._build_system_message(system_prompt, accept_usernames))

        return messages, user_warnings

    async def _process_message_content(
        self,
        curr_msg: discord.Message,
        curr_node: AskOLMoMsgNode,
        bot_user: discord.ClientUser,
    ):
        cleaned_content = curr_msg.content.removeprefix(bot_user.mention).lstrip()

        good_attachments = [
            att
            for att in curr_msg.attachments
            if att.content_type
            and any(att.content_type.startswith(x) for x in ("text", "image"))
        ]

        attachment_responses = await asyncio.gather(
            *[self.httpx_client.get(att.url) for att in good_attachments]
        )

        curr_node.text = "\n".join(
            ([cleaned_content] if cleaned_content else [])
            + [
                "\n".join(
                    filter(None, (embed.title, embed.description, embed.footer.text))
                )
                for embed in curr_msg.embeds
            ]
            + [
                resp.text
                for att, resp in zip(good_attachments, attachment_responses)
                if att.content_type.startswith("text")  # type: ignore
            ]
        )

        curr_node.images = [
            dict(
                type="image_url",
                image_url=dict(
                    url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"
                ),
            )
            for att, resp in zip(good_attachments, attachment_responses)
            if att.content_type.startswith("image")  # type: ignore
        ]

        curr_node.role = "assistant" if curr_msg.author == bot_user else "user"
        curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
        curr_node.has_bad_attachments = len(curr_msg.attachments) > len(
            good_attachments
        )

        await self._find_parent_message(curr_msg, curr_node, bot_user)

    async def _find_parent_message(
        self,
        curr_msg: discord.Message,
        curr_node: AskOLMoMsgNode,
        bot_user: discord.ClientUser,
    ):
        try:
            if (
                curr_msg.reference is None
                and bot_user.mention not in curr_msg.content
                and (
                    prev_msg_in_channel := (
                        [
                            m
                            async for m in curr_msg.channel.history(
                                before=curr_msg, limit=1
                            )
                        ]
                        or [None]
                    )[0]
                )
                and prev_msg_in_channel.type
                in (discord.MessageType.default, discord.MessageType.reply)
                and prev_msg_in_channel.author
                == (
                    bot_user
                    if curr_msg.channel.type == discord.ChannelType.private
                    else curr_msg.author
                )
            ):
                curr_node.parent_msg = prev_msg_in_channel
            else:
                is_public_thread = (
                    curr_msg.channel.type == discord.ChannelType.public_thread
                )
                parent_is_thread_start = (
                    is_public_thread
                    and curr_msg.reference is None
                    and curr_msg.channel.parent.type == discord.ChannelType.text  # type: ignore
                )

                if parent_msg_id := (
                    curr_msg.channel.id
                    if parent_is_thread_start
                    else getattr(curr_msg.reference, "message_id", None)
                ):
                    if parent_is_thread_start:
                        curr_node.parent_msg = (
                            curr_msg.channel.starter_message  # type: ignore
                            or await curr_msg.channel.parent.fetch_message(  # type: ignore
                                parent_msg_id
                            )
                        )
                    else:
                        curr_node.parent_msg = (
                            curr_msg.reference.cached_message  # type: ignore
                            or await curr_msg.channel.fetch_message(parent_msg_id)
                        )

        except (discord.NotFound, discord.HTTPException):
            logging.exception("Error fetching next message in the chain")
            curr_node.fetch_parent_failed = True

    def _build_message_content(
        self, curr_node: AskOLMoMsgNode, max_text: int, max_images: int
    ):
        if curr_node.images[:max_images]:
            content = (
                [dict(type="text", text=curr_node.text[:max_text])]  # type: ignore
                if curr_node.text[:max_text]  # type: ignore
                else []
            ) + curr_node.images[:max_images]
        else:
            content = curr_node.text[:max_text]  # type: ignore
        return content

    def _collect_warnings(
        self,
        curr_node: AskOLMoMsgNode,
        max_text: int,
        max_images: int,
        messages_len: int,
        max_messages: int,
        user_warnings: set,
    ):
        if len(curr_node.text) > max_text:  # type: ignore
            user_warnings.add(f"Max {max_text:,} characters per message")
        if len(curr_node.images) > max_images:
            user_warnings.add(
                f"Max {max_images} image{'' if max_images == 1 else 's'} per message"
                if max_images > 0
                else "Can't see images"
            )
        if curr_node.has_bad_attachments:
            user_warnings.add("Unsupported attachments")
        if curr_node.fetch_parent_failed or (
            curr_node.parent_msg is not None and messages_len == max_messages
        ):
            user_warnings.add(
                f"Only using last {messages_len} message{'' if messages_len == 1 else 's'}"
            )

    def _build_system_message(self, system_prompt: str, accept_usernames: bool) -> dict:
        now = datetime.now().astimezone()
        system_prompt = (
            system_prompt.replace("{date}", now.strftime("%B %d %Y"))
            .replace("{time}", now.strftime("%H:%M:%S %Z%z"))
            .strip()
        )
        # if accept_usernames:
        #     system_prompt += (
        #         "\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        #     )

        return dict(role="system", content=system_prompt)

    async def cleanup_message_nodes(self):
        if (num_nodes := len(self.msg_nodes)) > MAX_MESSAGE_NODES:
            for msg_id in sorted(self.msg_nodes.keys())[
                : num_nodes - MAX_MESSAGE_NODES
            ]:
                node = self.msg_nodes.get(msg_id)
                if node:
                    async with node.lock:
                        self.msg_nodes.pop(msg_id, None)

    def store_response_nodes(
        self,
        response_msgs: list[discord.Message],
        response_contents: list[str],
        new_msg: discord.Message,
    ):
        for response_msg in response_msgs:
            if response_msg.id in self.msg_nodes:
                self.msg_nodes[response_msg.id].text = "".join(response_contents)
                self.msg_nodes[response_msg.id].lock.release()


class AskOLMo:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_manager = AskOLMoConfigManager(config_file)
        self.config = self.config_manager.config
        self.curr_model = next(iter(self.config["models"]))

        self.permission_manager = AskOLMoPermissionManager(self.config)
        self.rate_limiter = AskOLMoRateLimiter(self.config)
        self.message_processor = AskOLMoMessageProcessor(self.config)
        self.response_generator = AskOLMoResponseGenerator(self.config, self.message_processor)

        try:
            self.safety_checker = GoogleModerateText()
            logging.info("Google moderation initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Google moderation: {e}")
            self.safety_checker = None

        self._setup_discord_bot()

    def _setup_discord_bot(self):
        intents = discord.Intents.default()
        intents.message_content = True
        activity = discord.CustomActivity(name=(self.config["status_message"])[:128])
        self.discord_bot = commands.Bot(
            intents=intents, activity=activity, command_prefix=None  # type: ignore
        )

        self._register_commands()
        self._register_events()

    def _register_commands(self):
        @self.discord_bot.tree.command(
            name="model", description="View or switch the current model"
        )
        async def model_command(interaction: discord.Interaction, model: str) -> None:
            await self.handle_model_command(interaction, model)

        @model_command.autocomplete("model")
        async def model_autocomplete(
            interaction: discord.Interaction, curr_str: str
        ) -> list[Choice[str]]:
            return await self.handle_model_autocomplete(interaction, curr_str)

    def _register_events(self):
        @self.discord_bot.event
        async def on_ready() -> None:
            await self.handle_ready()

        @self.discord_bot.event
        async def on_message(new_msg: discord.Message) -> None:
            await self.handle_message(new_msg)

        @self.discord_bot.event
        async def on_guild_join(guild: discord.Guild) -> None:
            await self.handle_guild_join(guild)

    async def handle_model_command(self, interaction: discord.Interaction, model: str):
        if model == self.curr_model:
            output = f"Current model: `{self.curr_model}`"
        else:
            if self.permission_manager.is_admin(interaction.user.id):
                self.curr_model = model
                output = f"Model switched to: `{model}`"
                logging.info(output)
            else:
                output = "You don't have permission to change the model."

        await interaction.response.send_message(
            output, ephemeral=(interaction.channel.type == discord.ChannelType.private)  # type: ignore
        )

    async def handle_model_autocomplete(
        self, interaction: discord.Interaction, curr_str: str
    ) -> list[Choice[str]]:
        if curr_str == "":
            config = await self.config_manager.reload_config()
        else:
            config = self.config

        choices = [
            Choice(name=f"○ {model}", value=model)
            for model in config["models"]
            if model != self.curr_model and curr_str.lower() in model.lower()
        ][:24]
        choices += (
            [Choice(name=f"◉ {self.curr_model} (current)", value=self.curr_model)]
            if curr_str.lower() in self.curr_model.lower()
            else []
        )

        return choices

    async def handle_ready(self):
        if client_id := self.config["client_id"]:
            logging.info(
                f"\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n"
            )

        await self.discord_bot.tree.sync()
        asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        while True:
            await asyncio.sleep(3600)
            try:
                self.rate_limiter.cleanup_old_data()
                logging.info("Rate limiter cleanup completed")
            except Exception as e:
                logging.error(f"Error during rate limiter cleanup: {e}")

    async def handle_message(self, new_msg: discord.Message):
        is_dm = new_msg.channel.type == discord.ChannelType.private

        if (
            not is_dm and self.discord_bot.user not in new_msg.mentions
        ) or new_msg.author.bot:
            return

        role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
        channel_ids = set(
            filter(
                None,
                (
                    new_msg.channel.id,
                    getattr(new_msg.channel, "parent_id", None),
                    getattr(new_msg.channel, "category_id", None),
                ),
            )
        )

        config = await self.config_manager.reload_config()
        self.permission_manager = AskOLMoPermissionManager(config)

        if not self.permission_manager.check_user_permissions(
            new_msg.author, role_ids, is_dm  # type: ignore
        ):
            return

        if not self.permission_manager.check_channel_permissions(
            new_msg.channel, channel_ids, is_dm, new_msg.author.id, config["allow_dms"]
        ):
            return

        if not self.permission_manager.is_admin(new_msg.author.id):
            if self.rate_limiter.is_rate_limited(new_msg.author.id):
                remaining_messages = self.rate_limiter.get_remaining_messages(new_msg.author.id)
                embed = discord.Embed(
                    title="Rate Limit Exceeded",
                    description=f"You have reached the maximum of {self.rate_limiter.max_messages_per_hour} messages per hour. Please wait before sending another message.",
                    color=discord.Color.red()
                )
                embed.add_field(
                    name="Ai2 Playground",
                    value="Visit the [Ai2 Playground](https://playground.allenai.org) to continue interacting with our models.",
                    inline=False
                )
                embed.set_footer(text="Rate limit resets every hour from your first message.")
                
                await new_msg.reply(embed=embed, silent=True)
                return
            
            self.rate_limiter.add_message(new_msg.author.id)

        if self.safety_checker and new_msg.content.strip():
            try:
                safety_request = SafetyCheckRequest(content=new_msg.content)
                safety_response = self.safety_checker.check_request(safety_request)

                # if not safety_response.is_safe():
                #     violations = safety_response.get_violation_categories()
                #     logging.warning(f"Unsafe message detected from user {new_msg.author.id}: {violations}")

                #     embed = discord.Embed(
                #         title="Content Warning",
                #         description="Your message was flagged by our content moderation system and cannot be processed.",
                #         color=discord.Color.red()
                #     )
                #     # embed.add_field(
                #     #     name="Violations Detected",
                #     #     value="\n".join(violations[:3]),  # Show up to 3 violations
                #     #     inline=False
                #     # )
                #     # embed.set_footer(text="Please modify your message to comply with our content guidelines.")

                #     await new_msg.reply(embed=embed, silent=True)
                #     return

            except Exception as e:
                logging.error("Error during safety check: %s", e)

        provider_slash_model = self.curr_model
        model_parameters = config["models"].get(provider_slash_model, None)

        messages, user_warnings = await self.message_processor.process_message_chain(
            new_msg, self.curr_model, self.discord_bot.user  # type: ignore
        )

        # logging.info(
        #     f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}"
        # )

        response_msgs, response_contents = (
            await self.response_generator.generate_response(
                messages, self.curr_model, model_parameters, new_msg
            )
        )

        embed = discord.Embed()
        for warning in sorted(user_warnings):
            embed.add_field(name=warning, value="", inline=False)

        if user_warnings and response_msgs:
            warning_embed = discord.Embed()
            for warning in sorted(user_warnings):
                warning_embed.add_field(name=warning, value="", inline=False)
            try:
                await response_msgs[0].edit(embed=warning_embed)
            except:
                pass

        self.message_processor.store_response_nodes(
            response_msgs, response_contents, new_msg
        )
        await self.message_processor.cleanup_message_nodes()

    async def handle_guild_join(self, guild: discord.Guild):
        """Handle when the bot joins a new guild - leave if not authorized"""
        allowed_guilds = self.config.get("allowed_guild_ids", [])
        
        if allowed_guilds and guild.id not in allowed_guilds:
            logging.warning(f"Bot added to unauthorized guild {guild.name} ({guild.id}). Leaving...")
            try:
                await guild.leave()
                logging.info(f"Successfully left unauthorized guild {guild.name}")
            except Exception as e:
                logging.error(f"Failed to leave guild {guild.name}: {e}")
        else:
            logging.info(f"Bot joined authorized guild: {guild.name} ({guild.id})")

    async def start(self):
        await self.discord_bot.start(self.config["bot_token"])


async def main() -> None:
    bot = AskOLMo()
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
