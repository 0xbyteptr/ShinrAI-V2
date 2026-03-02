#!/usr/bin/env python3
"""
Shinrai Discord Bot - Chat with your AI in Discord
"""

import discord
from discord.ext import commands
from discord import app_commands
from discord.ext import tasks
import asyncio
import logging
import json
import os
from pathlib import Path
import sys
import random
from datetime import datetime, timedelta
import hashlib

# Add Shinrai to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shinrai.core import Shinrai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
CONFIG_FILE = "discord_config.json"
DEFAULT_CONFIG = {
    "token": "YOUR_DISCORD_BOT_TOKEN_HERE",
    "command_prefix": "!",
    "model_path": "shinrai_model",
    "max_history": 10,
    "response_timeout": 30,
    "allow_dm": True,
    "channels": [],
    "cooldown": 3,
    "max_message_length": 1900,
    "status_update_seconds": 45,
    "sync_slash_commands": True,
    "auto_learn_from_channels": True,
    "auto_learn_channels": [],
    "auto_learn_min_chars": 20,
    "auto_learn_batch_size": 20,
    "auto_learn_save_every_batches": 1,
    "auto_learn_update_topics": False,
    "hf_default_rows": 2000,
    # intents control what events the bot can receive.  By default we use
    # the *minimum* intents needed to respond to messages.  Enabling
    # message_content or members/presences requires you to turn on the
    # corresponding privileged intent in your Discord developer portal.
    "intents": {
        "message_content": False,
        "members": False,
        "presences": False
    }
}

class ShinraiDiscordBot:
    """Discord bot for Shinrai AI"""
    
    def __init__(self):
        self.config = self.load_config()
        # build intents according to configuration, falling back to a
        # safe default that avoids privileged intents.  ``message_content``
        # also counts as privileged and must be explicitly enabled in the
        # developer portal if set to True.
        intents_cfg = self.config.get('intents', {})
        intents = discord.Intents.default()
        if intents_cfg.get('message_content'):
            intents.message_content = True
        if intents_cfg.get('members'):
            intents.members = True
        if intents_cfg.get('presences'):
            intents.presences = True

        self.bot = commands.Bot(
            command_prefix=self.config['command_prefix'],
            intents=intents,
            help_command=None
        )
        
        # Initialize Shinrai
        logger.info("Initializing Shinrai AI...")
        self.shinrai = Shinrai(model_path=self.config['model_path'])
        # warn about privileged intents that may not be enabled
        if intents_cfg.get('message_content') and not intents.message_content:
            logger.warning("message_content intent requested but not enabled; check your Discord developer portal")
        if self.config.get('auto_learn_from_channels') and not intents.message_content:
            logger.warning("auto_learn_from_channels is enabled but message_content intent is disabled")
        
        # Conversation memory per user/channel
        self.conversations = {}
        self.cooldowns = {}
        self.shinrai_lock = asyncio.Lock()
        self.learning_lock = asyncio.Lock()
        self.learning_buffer = []
        self.learned_batches = 0
        self.learned_messages = 0
        # persisted conversations file
        self.conv_file = "conversations.json"
        self._load_conversations()
        
        # Setup bot events and commands
        self.setup_events()
        self.setup_commands()
        self.status_loop.change_interval(seconds=max(15, int(self.config.get('status_update_seconds', 45))))
        
        # Stats
        self.stats = {
            'messages_processed': 0,
            'commands_used': 0,
            'slash_commands_used': 0,
            'ai_responses_generated': 0,
            'users_served': set(),
            'start_time': datetime.now(),
            'last_message_at': None,
            'learned_messages': 0,
            'learned_batches': 0
        }
    
    def _load_conversations(self):
        """Load persisted conversation history from disk"""
        if os.path.exists(self.conv_file):
            try:
                with open(self.conv_file, 'r') as f:
                    self.conversations = json.load(f)
                logger.info(f"Loaded {len(self.conversations)} conversations from disk")
            except Exception as e:
                logger.warning(f"Could not load conversations: {e}")
        
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            # ensure new keys exist (migration)
            changed = False
            if 'intents' not in cfg:
                cfg['intents'] = DEFAULT_CONFIG['intents']
                changed = True
            # if any individual intent key is missing, add it
            for intent_key, default_val in DEFAULT_CONFIG['intents'].items():
                if intent_key not in cfg.get('intents', {}):
                    cfg['intents'][intent_key] = default_val
                    changed = True
            # ensure any new top-level config keys are present
            for key, default_val in DEFAULT_CONFIG.items():
                if key == 'intents':
                    continue
                if key not in cfg:
                    cfg[key] = default_val
                    changed = True
            if changed:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(cfg, f, indent=4)
                logger.info("Updated config file with new intent settings")
            return cfg
        else:
            # Create default config
            with open(CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            logger.info(f"Created default config file: {CONFIG_FILE}")
            logger.info("Please edit the file and add your bot token")
            return DEFAULT_CONFIG
    
    def save_config(self):
        """Save configuration to file"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def setup_events(self):
        """Set up bot events"""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Logged in as {self.bot.user.name}")
            logger.info(f"Bot ID: {self.bot.user.id}")
            logger.info(f"Shinrai knowledge base: {len(self.shinrai.documents)} documents")
            logger.info("------")

            if self.config.get('sync_slash_commands', True):
                try:
                    synced = await self.bot.tree.sync()
                    logger.info(f"Synced {len(synced)} slash commands")
                except Exception as e:
                    logger.warning(f"Failed to sync slash commands: {e}")

            await self.refresh_presence()
            if not self.status_loop.is_running():
                self.status_loop.start()
        
        @self.bot.event
        async def on_message(message):
            # Ignore bot messages
            if message.author.bot:
                return

            await self.maybe_learn_from_message(message)
            
            # Process commands first
            await self.bot.process_commands(message)
            
            # Check if we should respond to normal messages
            if await self.should_respond(message):
                await self.handle_conversation(message)
        
        @self.bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.CommandOnCooldown):
                await ctx.send(f"⏰ Command on cooldown. Try again in {error.retry_after:.1f}s")
            elif isinstance(error, commands.MissingPermissions):
                await ctx.send("❌ You don't have permission to use this command.")
            else:
                logger.error(f"Command error: {error}")
                await ctx.send(f"❌ An error occurred: {str(error)}")
    
    def setup_commands(self):
        """Register prefix and slash commands."""
        for cmd in list(self.bot.commands):
            try:
                self.bot.remove_command(cmd.name)
            except Exception:
                pass

        self.bot.tree.clear_commands(guild=None)

        @self.bot.command(name="chat", aliases=["ask", "c"])
        @commands.cooldown(1, 3, commands.BucketType.user)
        async def chat(ctx, *, message):
            async with ctx.typing():
                response = await self.get_ai_response(message, ctx)
                await self.send_long_message(ctx, response)
            self.stats['commands_used'] += 1

        @self.bot.command(name="help", aliases=["h", "commands"])
        async def help_command(ctx):
            embed = discord.Embed(
                title="🤖 Shinrai Commands",
                description="Prefix and slash commands are both enabled.",
                color=discord.Color.blue()
            )
            commands_list = [
                (f"{self.config['command_prefix']}chat <message>", "Chat with Shinrai"),
                (f"{self.config['command_prefix']}stats", "Bot usage and runtime stats"),
                (f"{self.config['command_prefix']}status", "Live bot status"),
                (f"{self.config['command_prefix']}info", "Model and knowledge info"),
                (f"{self.config['command_prefix']}memory", "Show your recent conversation"),
                (f"{self.config['command_prefix']}learn-status", "Show auto-learning queue/status"),
                (f"{self.config['command_prefix']}learn-flush", "Flush learned messages (admin)"),
                ("/chat", "Chat via slash command"),
                ("/stats", "Stats via slash command"),
                ("/status", "Status via slash command"),
                ("/clear", "Clear your memory"),
                ("/learn_status", "Show auto-learning status")
            ]
            for cmd, desc in commands_list:
                embed.add_field(name=cmd, value=desc, inline=False)
            await ctx.send(embed=embed)

        @self.bot.command(name="clear", aliases=["reset"])
        async def clear_memory(ctx):
            user_id = str(ctx.author.id)
            if user_id in self.conversations:
                del self.conversations[user_id]
                self._save_conversations()
            await ctx.send("🧹 Your conversation history was cleared.")
            self.stats['commands_used'] += 1

        @self.bot.command(name="stats")
        async def show_stats(ctx):
            await ctx.send(embed=self._build_stats_embed())
            self.stats['commands_used'] += 1

        @self.bot.command(name="status")
        async def check_status(ctx):
            await ctx.send(embed=self._build_status_embed())
            self.stats['commands_used'] += 1

        @self.bot.command(name="info")
        async def show_info(ctx):
            await ctx.send(embed=self._build_info_embed())
            self.stats['commands_used'] += 1

        @self.bot.command(name="summarize")
        async def summarize_knowledge(ctx):
            summary = self.shinrai.response_generator._summarize_knowledge(self.shinrai.knowledge_graph)
            await self.send_long_message(ctx, summary)
            self.stats['commands_used'] += 1

        @self.bot.command(name="memory")
        async def show_memory(ctx):
            user_id = str(ctx.author.id)
            if user_id not in self.conversations or not self.conversations[user_id]:
                await ctx.send("📭 No conversation history found.")
                return
            history = self.conversations[user_id][-5:]
            embed = discord.Embed(title="🧠 Your Recent Conversation", color=discord.Color.blue())
            for i, msg in enumerate(history, 1):
                embed.add_field(
                    name=f"Message {i}",
                    value=f"**You:** {msg['user'][:80]}\n**Bot:** {msg['bot'][:120]}",
                    inline=False
                )
            await ctx.send(embed=embed)
            self.stats['commands_used'] += 1

        @self.bot.command(name="learn-status")
        async def learn_status(ctx):
            embed = discord.Embed(title="🧠 Auto Learning Status", color=discord.Color.teal())
            embed.add_field(name="Enabled", value=str(self.config.get('auto_learn_from_channels', False)), inline=True)
            embed.add_field(name="Queue Size", value=str(len(self.learning_buffer)), inline=True)
            embed.add_field(name="Batch Size", value=str(self.config.get('auto_learn_batch_size', 20)), inline=True)
            embed.add_field(name="Learned Messages", value=str(self.stats['learned_messages']), inline=True)
            embed.add_field(name="Learned Batches", value=str(self.stats['learned_batches']), inline=True)
            allowed = self.config.get('auto_learn_channels', []) or []
            embed.add_field(name="Channel Scope", value="All" if not allowed else str(len(allowed)), inline=True)
            await ctx.send(embed=embed)
            self.stats['commands_used'] += 1

        @self.bot.command(name="learn-flush")
        @commands.has_permissions(administrator=True)
        async def learn_flush(ctx):
            queued = len(self.learning_buffer)
            if queued == 0:
                await ctx.send("ℹ️ Learning queue is empty.")
            else:
                await ctx.send(f"🔄 Flushing learning queue ({queued} messages)...")
                await self.flush_learning_buffer()
                await ctx.send("✅ Learning queue flushed and saved.")
            self.stats['commands_used'] += 1

        @self.bot.command(name="train")
        @commands.has_permissions(administrator=True)
        async def train_command(ctx, url: str, pages: int = 10):
            source_type, mode_kwargs, amount = self._resolve_train_mode(url, pages)
            unit = "rows" if source_type == 'hf_dataset' else "pages"
            await ctx.send(f"🔄 Training on {url} (max {amount} {unit}). This may take a while...")
            async with ctx.typing():
                try:
                    loop = asyncio.get_event_loop()
                    async with self.shinrai_lock:
                        await loop.run_in_executor(
                            None,
                            lambda: self.shinrai.train(url, source_type=source_type, **mode_kwargs)
                        )
                    embed = discord.Embed(title="✅ Training Complete", color=discord.Color.green())
                    embed.add_field(name="URL", value=url, inline=False)
                    embed.add_field(name=unit.capitalize(), value=str(amount), inline=True)
                    embed.add_field(name="Source Type", value=source_type, inline=True)
                    embed.add_field(name="Total Documents", value=str(len(self.shinrai.documents)), inline=True)
                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Training failed: {str(e)}")
            self.stats['commands_used'] += 1

        @self.bot.command(name="config")
        @commands.has_permissions(administrator=True)
        async def show_config(ctx):
            embed = discord.Embed(title="⚙️ Bot Configuration", color=discord.Color.orange())
            for key, value in self.config.items():
                if key != "token":
                    embed.add_field(name=key, value=str(value), inline=False)
            await ctx.send(embed=embed)
            self.stats['commands_used'] += 1

        @self.bot.command(name="reload-config")
        @commands.has_permissions(administrator=True)
        async def reload_config(ctx):
            self.config = self.load_config()
            await ctx.send("🔄 Configuration reloaded.")
            self.stats['commands_used'] += 1

        @self.bot.tree.command(name="chat", description="Chat with Shinrai")
        async def slash_chat(interaction: discord.Interaction, message: str):
            await interaction.response.defer(thinking=True)
            response = await self.get_ai_response(message, interaction)
            await self._send_long_interaction(interaction, response)
            self.stats['slash_commands_used'] += 1

        @self.bot.tree.command(name="stats", description="Show bot stats")
        async def slash_stats(interaction: discord.Interaction):
            await interaction.response.send_message(embed=self._build_stats_embed())
            self.stats['slash_commands_used'] += 1

        @self.bot.tree.command(name="status", description="Show live bot status")
        async def slash_status(interaction: discord.Interaction):
            await interaction.response.send_message(embed=self._build_status_embed())
            self.stats['slash_commands_used'] += 1

        @self.bot.tree.command(name="info", description="Show model and corpus info")
        async def slash_info(interaction: discord.Interaction):
            await interaction.response.send_message(embed=self._build_info_embed())
            self.stats['slash_commands_used'] += 1

        @self.bot.tree.command(name="clear", description="Clear your conversation history")
        async def slash_clear(interaction: discord.Interaction):
            user_id = str(interaction.user.id)
            if user_id in self.conversations:
                del self.conversations[user_id]
                self._save_conversations()
            await interaction.response.send_message("🧹 Your conversation history was cleared.", ephemeral=True)
            self.stats['slash_commands_used'] += 1

        @self.bot.tree.command(name="memory", description="Show your recent conversation")
        async def slash_memory(interaction: discord.Interaction):
            user_id = str(interaction.user.id)
            if user_id not in self.conversations or not self.conversations[user_id]:
                await interaction.response.send_message("📭 No conversation history found.", ephemeral=True)
                return
            history = self.conversations[user_id][-5:]
            embed = discord.Embed(title="🧠 Your Recent Conversation", color=discord.Color.blue())
            for i, msg in enumerate(history, 1):
                embed.add_field(name=f"Message {i}", value=f"**You:** {msg['user'][:80]}\n**Bot:** {msg['bot'][:120]}", inline=False)
            await interaction.response.send_message(embed=embed, ephemeral=True)
            self.stats['slash_commands_used'] += 1

        @self.bot.tree.command(name="summarize", description="Summarize learned knowledge")
        async def slash_summarize(interaction: discord.Interaction):
            summary = self.shinrai.response_generator._summarize_knowledge(self.shinrai.knowledge_graph)
            await interaction.response.send_message(summary)
            self.stats['slash_commands_used'] += 1

        @self.bot.tree.command(name="learn_status", description="Show channel-learning status")
        async def slash_learn_status(interaction: discord.Interaction):
            embed = discord.Embed(title="🧠 Auto Learning Status", color=discord.Color.teal())
            embed.add_field(name="Enabled", value=str(self.config.get('auto_learn_from_channels', False)), inline=True)
            embed.add_field(name="Queue Size", value=str(len(self.learning_buffer)), inline=True)
            embed.add_field(name="Batch Size", value=str(self.config.get('auto_learn_batch_size', 20)), inline=True)
            embed.add_field(name="Learned Messages", value=str(self.stats['learned_messages']), inline=True)
            embed.add_field(name="Learned Batches", value=str(self.stats['learned_batches']), inline=True)
            await interaction.response.send_message(embed=embed, ephemeral=True)
            self.stats['slash_commands_used'] += 1

        @app_commands.default_permissions(administrator=True)
        @self.bot.tree.command(name="learn_flush", description="Force flush learned-channel messages")
        async def slash_learn_flush(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True, thinking=True)
            queued = len(self.learning_buffer)
            if queued == 0:
                await interaction.followup.send("ℹ️ Learning queue is empty.")
            else:
                await self.flush_learning_buffer()
                await interaction.followup.send(f"✅ Flushed {queued} queued messages.")
            self.stats['slash_commands_used'] += 1

        @app_commands.default_permissions(administrator=True)
        @self.bot.tree.command(name="train", description="Train on a URL (admin)")
        async def slash_train(interaction: discord.Interaction, url: str, pages: int = 10):
            await interaction.response.defer(thinking=True)
            try:
                source_type, mode_kwargs, amount = self._resolve_train_mode(url, pages)
                unit = "rows" if source_type == 'hf_dataset' else "pages"
                loop = asyncio.get_event_loop()
                async with self.shinrai_lock:
                    await loop.run_in_executor(
                        None,
                        lambda: self.shinrai.train(url, source_type=source_type, **mode_kwargs)
                    )
                embed = discord.Embed(title="✅ Training Complete", color=discord.Color.green())
                embed.add_field(name="URL", value=url, inline=False)
                embed.add_field(name=unit.capitalize(), value=str(amount), inline=True)
                embed.add_field(name="Source Type", value=source_type, inline=True)
                embed.add_field(name="Total Documents", value=str(len(self.shinrai.documents)), inline=True)
                await interaction.followup.send(embed=embed)
            except Exception as e:
                await interaction.followup.send(f"❌ Training failed: {e}")
            self.stats['slash_commands_used'] += 1

    def _save_conversations(self):
        """Persist conversations dict to disk"""
        try:
            with open(self.conv_file, 'w') as f:
                json.dump(self.conversations, f)
        except Exception as e:
            logger.warning(f"Failed to save conversations: {e}")

    def _format_uptime(self) -> str:
        uptime = datetime.now() - self.stats['start_time']
        return str(uptime).split('.')[0]

    def _get_parameter_count(self) -> int:
        model = getattr(self.shinrai, 'transformer_model', None)
        if model is None:
            return 0
        try:
            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                return sum(p.numel() for p in model.model.parameters())
        except Exception:
            return 0
        return 0

    def _build_stats_embed(self) -> discord.Embed:
        embed = discord.Embed(title="📊 Shinrai Bot Statistics", color=discord.Color.gold())
        embed.add_field(name="Messages Processed", value=str(self.stats['messages_processed']), inline=True)
        embed.add_field(name="AI Responses", value=str(self.stats['ai_responses_generated']), inline=True)
        embed.add_field(name="Users Served", value=str(len(self.stats['users_served'])), inline=True)
        embed.add_field(name="Prefix Commands", value=str(self.stats['commands_used']), inline=True)
        embed.add_field(name="Slash Commands", value=str(self.stats['slash_commands_used']), inline=True)
        embed.add_field(name="Uptime", value=self._format_uptime(), inline=True)
        embed.add_field(name="Knowledge Base", value=str(len(self.shinrai.documents)), inline=True)
        embed.add_field(name="Active Conversations", value=str(len(self.conversations)), inline=True)
        embed.add_field(name="Latency", value=f"{round(self.bot.latency * 1000)}ms", inline=True)
        embed.add_field(name="Learned Messages", value=str(self.stats['learned_messages']), inline=True)
        embed.add_field(name="Learned Batches", value=str(self.stats['learned_batches']), inline=True)
        embed.add_field(name="Learning Queue", value=str(len(self.learning_buffer)), inline=True)
        return embed

    def _build_status_embed(self) -> discord.Embed:
        param_count = self._get_parameter_count()
        embed = discord.Embed(title="✅ Bot Status", color=discord.Color.green())
        embed.add_field(name="Online", value="Yes", inline=True)
        embed.add_field(name="Latency", value=f"{round(self.bot.latency * 1000)}ms", inline=True)
        embed.add_field(name="Documents", value=str(len(self.shinrai.documents)), inline=True)
        embed.add_field(name="Model Loaded", value="Yes" if self.shinrai.embeddings is not None else "Partial", inline=True)
        embed.add_field(name="Encoder Params", value=f"{param_count:,}" if param_count else "Unavailable", inline=True)
        embed.add_field(name="Uptime", value=self._format_uptime(), inline=True)
        embed.add_field(
            name="Auto Learn",
            value="Enabled" if self.config.get('auto_learn_from_channels') else "Disabled",
            inline=True
        )
        embed.add_field(name="Learned", value=str(self.stats['learned_messages']), inline=True)
        return embed

    def _build_info_embed(self) -> discord.Embed:
        embed = discord.Embed(
            title="🤖 About Shinrai AI",
            description="An uncensored AI chatbot that learns from websites",
            color=discord.Color.purple()
        )
        embed.add_field(name="Model Path", value=self.config['model_path'], inline=False)
        embed.add_field(name="Documents", value=str(len(self.shinrai.documents)), inline=True)
        embed.add_field(name="Embeddings", value="Loaded" if self.shinrai.embeddings is not None else "Not loaded", inline=True)
        embed.add_field(name="Device", value=str(getattr(self.shinrai, 'device', 'unknown')), inline=True)
        embed.add_field(name="Features", value="Web scraping, Knowledge graph, Conversation memory", inline=False)
        return embed

    def _resolve_train_mode(self, url: str, pages: int):
        """Resolve training source and kwargs from /train arguments.

        For Hugging Face dataset URLs, `pages` is interpreted as max rows.
        If user keeps the default value (10), we switch to hf_default_rows.
        """
        is_hf = self.shinrai._looks_like_hf_dataset_source(url)
        if is_hf:
            max_rows = pages
            if pages == 10:
                max_rows = int(self.config.get('hf_default_rows', 2000))
            return 'hf_dataset', {'max_rows': max_rows}, max_rows

        return 'web', {'max_pages': pages}, pages

    async def refresh_presence(self):
        status_text = (
            f"{self.stats['messages_processed']} msgs • "
            f"{len(self.shinrai.documents)} docs • "
            f"{self.stats['learned_messages']} learned"
        )
        await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=status_text))

    @tasks.loop(seconds=30)
    async def status_loop(self):
        await self.refresh_presence()

    @status_loop.before_loop
    async def before_status_loop(self):
        await self.bot.wait_until_ready()

    async def _send_long_interaction(self, interaction: discord.Interaction, content: str):
        if len(content) <= self.config['max_message_length']:
            await interaction.followup.send(content)
            return

        chunks = []
        current_chunk = ""
        for sentence in content.split('. '):
            if len(current_chunk) + len(sentence) + 2 < self.config['max_message_length']:
                current_chunk = f"{current_chunk}. {sentence}" if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk + '.')
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk + '.')

        for chunk in chunks:
            await interaction.followup.send(chunk)

    def _should_learn_message(self, message: discord.Message) -> bool:
        if not self.config.get('auto_learn_from_channels', False):
            return False
        if isinstance(message.channel, discord.DMChannel):
            return False
        if not message.content:
            return False

        content = message.content.strip()
        if len(content) < int(self.config.get('auto_learn_min_chars', 20)):
            return False

        # ignore commands and direct mentions to avoid polluting KB
        prefix = str(self.config.get('command_prefix', '!'))
        if content.startswith(prefix) or content.startswith('/'):
            return False
        if self.bot.user and self.bot.user.mention in content:
            return False

        allowed_channels = self.config.get('auto_learn_channels', []) or []
        if allowed_channels and message.channel.id not in allowed_channels:
            return False

        return True

    async def maybe_learn_from_message(self, message: discord.Message):
        if not self._should_learn_message(message):
            return

        sample = {
            'guild': getattr(message.guild, 'name', 'DM') if message.guild else 'DM',
            'channel': getattr(message.channel, 'name', str(message.channel.id)),
            'channel_id': message.channel.id,
            'author': str(message.author),
            'author_id': message.author.id,
            'content': message.content.strip(),
            'url': message.jump_url,
            'timestamp': message.created_at.isoformat() if message.created_at else datetime.now().isoformat()
        }
        self.learning_buffer.append(sample)

        batch_size = max(1, int(self.config.get('auto_learn_batch_size', 20)))
        if len(self.learning_buffer) >= batch_size:
            await self.flush_learning_buffer()

    def _ingest_channel_samples_sync(self, samples):
        texts = []
        metadata = []

        for item in samples:
            content = ' '.join(item['content'].split())
            text = f"[discord:{item['guild']}/{item['channel']}] {item['author']}: {content}"
            texts.append(text)
            metadata.append({
                'source': 'discord',
                'guild': item['guild'],
                'channel': item['channel'],
                'channel_id': item['channel_id'],
                'author': item['author'],
                'author_id': item['author_id'],
                'url': item['url'],
                'timestamp': item['timestamp']
            })

        self.shinrai.documents.extend(texts)
        self.shinrai.document_metadata.extend(metadata)

        # embeddings are optional; keep learning functional even if encoder fails
        try:
            if self.shinrai.transformer_model is not None:
                self.shinrai._create_embeddings(texts, batch_size=min(32, max(4, len(texts))))
        except Exception as e:
            logger.warning(f"Failed to embed learned messages batch: {e}")

        try:
            self.shinrai._build_knowledge_graph(texts)
        except Exception as e:
            logger.warning(f"Failed to update knowledge graph from learned messages: {e}")

        if self.config.get('auto_learn_update_topics', False):
            try:
                self.shinrai._train_topic_model(self.shinrai.documents)
            except Exception as e:
                logger.warning(f"Failed to refresh topic model from learned messages: {e}")

        self.shinrai.save_model()
        return len(texts)

    async def flush_learning_buffer(self):
        if not self.learning_buffer:
            return

        async with self.learning_lock:
            if not self.learning_buffer:
                return

            samples = self.learning_buffer
            self.learning_buffer = []

            async with self.shinrai_lock:
                loop = asyncio.get_event_loop()
                learned_count = await loop.run_in_executor(None, self._ingest_channel_samples_sync, samples)

            self.learned_batches += 1
            self.learned_messages += learned_count
            self.stats['learned_batches'] = self.learned_batches
            self.stats['learned_messages'] = self.learned_messages
            logger.info(
                f"Learned {learned_count} channel messages (batch {self.learned_batches}); "
                f"total learned={self.learned_messages}"
            )
    
    async def should_respond(self, message):
        """Determine if bot should respond to a message"""
        # Check if DM and DMs are allowed
        if isinstance(message.channel, discord.DMChannel):
            return self.config['allow_dm']
        
        # Check if in allowed channels list
        if self.config['channels'] and message.channel.id not in self.config['channels']:
            return False
        
        # Check if bot is mentioned
        if self.bot.user in message.mentions:
            return True
        
        # Check if message starts with bot name
        if message.content.lower().startswith('shinrai'):
            return True
        
        return False
    
    async def handle_conversation(self, message):
        """Handle normal conversation messages"""
        # Check cooldown
        user_id = message.author.id
        if user_id in self.cooldowns:
            if datetime.now() < self.cooldowns[user_id]:
                return
            del self.cooldowns[user_id]
        
        # Remove bot mention from message
        content = message.content
        for mention in message.mentions:
            if mention.id == self.bot.user.id:
                content = content.replace(f'<@!{mention.id}>', '').replace(f'<@{mention.id}>', '').strip()
        
        if not content:
            return
        
        # Show typing indicator
        async with message.channel.typing():
            response = await self.get_ai_response(content, message)
            
            # Send response
            await self.send_long_message(message.channel, response, reference=message)
        
        # Set cooldown
        self.cooldowns[user_id] = datetime.now() + timedelta(seconds=self.config['cooldown'])
        
        # Update stats
        self.stats['messages_processed'] += 1
        self.stats['users_served'].add(user_id)
        self.stats['last_message_at'] = datetime.now().isoformat()
    
    async def get_ai_response(self, message: str, context) -> str:
        """Get response from Shinrai AI"""
        try:
            # Get conversation ID (user ID for DM, channel ID for guild)
            if isinstance(context, discord.Message):
                conv_id = str(context.author.id)
            elif isinstance(context, discord.Interaction):
                conv_id = str(context.user.id)
            else:
                conv_id = str(getattr(context, 'author', getattr(context, 'user')).id)
            
            # Get conversation history
            history = self.conversations.get(conv_id, [])
            
            loop = asyncio.get_event_loop()
            async with self.shinrai_lock:
                # Get relevant documents
                relevant_docs = await loop.run_in_executor(
                    None,
                    self.shinrai._get_relevant_documents,
                    message,
                    5
                )

                # Generate response
                response = await loop.run_in_executor(
                    None,
                    self.shinrai.response_generator.generate,
                    message,
                    relevant_docs,
                    self.shinrai.conversation_memory,
                    self.shinrai.knowledge_graph
                )
            
            # Store in conversation history
            history.append({
                'user': message,
                'bot': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last N messages
            if len(history) > self.config['max_history']:
                history = history[-self.config['max_history']:]
            
            self.conversations[conv_id] = history
            self._save_conversations()

            self.stats['ai_responses_generated'] += 1
            self.stats['users_served'].add(int(conv_id))
            self.stats['last_message_at'] = datetime.now().isoformat()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error: {str(e)}"
    
    async def send_long_message(self, destination, content, reference=None):
        """Send long messages by splitting them"""
        if len(content) <= self.config['max_message_length']:
            await destination.send(content, reference=reference)
        else:
            # Split into chunks
            chunks = []
            current_chunk = ""
            
            for sentence in content.split('. '):
                if len(current_chunk) + len(sentence) + 2 < self.config['max_message_length']:
                    if current_chunk:
                        current_chunk += '. ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk + '.')
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk + '.')
            
            # Send first chunk with reference, others without
            for i, chunk in enumerate(chunks):
                if i == 0 and reference:
                    await destination.send(chunk, reference=reference)
                else:
                    await destination.send(chunk)
                
                # Small delay between messages
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
    
    def run(self):
        """Run the bot"""
        token = self.config['token']
        if token == "YOUR_DISCORD_BOT_TOKEN_HERE":
            logger.error("Please set your Discord bot token in discord_config.json")
            logger.info("Get a token from: https://discord.com/developers/applications")
            return
        
        self.bot.run(token)

def main():
    """Main entry point"""
    bot = ShinraiDiscordBot()
    bot.run()

if __name__ == "__main__":
    main()