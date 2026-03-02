#!/usr/bin/env python3
"""
Shinrai Discord Bot - Chat with your AI in Discord
"""

import discord
from discord.ext import commands
import asyncio
import logging
import json
import os
from pathlib import Path
import sys
import random
from datetime import datetime, timedelta
import hashlib

from torch.serialization import _save

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
        
        # Conversation memory per user/channel
        self.conversations = {}
        self.cooldowns = {}
        # persisted conversations file
        self.conv_file = "conversations.json"
        self._load_conversations()
        
        # Setup bot events and commands
        self.setup_events()
        self.setup_commands()
        
        # Stats
        self.stats = {
            'messages_processed': 0,
            'users_served': set(),
            'start_time': datetime.now()
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
            
            # Set custom status
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.listening,
                    name=f"{self.config['command_prefix']}help"
                )
            )
        
        @self.bot.event
        async def on_message(message):
            # Ignore bot messages
            if message.author.bot:
                return
            
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
        """Set up bot commands.

        Discord.py raises an error if you try to add a command with the same
        name/alias twice.  This can happen if the initializer is invoked
        multiple times (e.g. during a hot reload), so we defensively remove
        any previously-registered commands before recreating them.
        """
        # strip out any earlier registrations that may exist
        for cmd in list(self.bot.commands):
            try:
                self.bot.remove_command(cmd.name)
            except Exception:
                pass

        # patch add_command so duplicate registrations don't crash
        orig_add = self.bot.add_command
        def safe_add(command, *args, **kwargs):
            try:
                return orig_add(command, *args, **kwargs)
            except Exception as e:
                # ignore 'already an existing command' errors
                if 'already an existing command' in str(e):
                    logger.debug(f"Ignored duplicate command: {command.name}")
                    return None
                raise
        self.bot.add_command = safe_add
    def _save_conversations(self):
        """Persist conversations dict to disk"""
        try:
            with open(self.conv_file, 'w') as f:
                json.dump(self.conversations, f)
        except Exception as e:
            logger.warning(f"Failed to save conversations: {e}")
        
        @self.bot.command(name="chat", aliases=["ask", "c"])
        @commands.cooldown(1, 3, commands.BucketType.user)
        async def chat(ctx, *, message):
            """Chat with Shinrai - Usage: !chat <your message>"""
            async with ctx.typing():
                response = await self.get_ai_response(message, ctx)
                
                # Split long messages
                await self.send_long_message(ctx, response)
            
            # Update stats
            self.stats['messages_processed'] += 1
            self.stats['users_served'].add(ctx.author.id)
        
        @self.bot.command(name="help", aliases=["h", "commands"])
        async def help_command(ctx):
            """Show available commands"""
            embed = discord.Embed(
                title="🤖 Shinrai AI Bot Commands",
                description="Chat with an AI that learns from websites!",
                color=discord.Color.blue()
            )
            
            commands_list = [
                ("!chat <message>", "Chat with Shinrai (or !ask, !c)"),
                ("!clear", "Clear your conversation history"),
                ("!stats", "Show bot statistics"),
                ("!info", "Show information about Shinrai"),
                ("!train <url>", "Train on a website (Admin only)"),
                ("!status", "Check bot status"),
                ("!memory", "Show your conversation memory"),
                ("!reset", "Reset your conversation")
            ]
            
            for cmd, desc in commands_list:
                embed.add_field(name=cmd, value=desc, inline=False)
            
            embed.set_footer(text=f"Knowledge base: {len(self.shinrai.documents)} documents")
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="clear", aliases=["reset"])
        async def clear_memory(ctx):
            """Clear your conversation history"""
            user_id = str(ctx.author.id)
            if user_id in self.conversations:
                del self.conversations[user_id]
            
            embed = discord.Embed(
                title="🧹 Memory Cleared",
                description="Your conversation history has been reset.",
                color=discord.Color.green()
            )
            await ctx.send(embed=embed)
        
        @self.bot.command(name="stats")
        async def show_stats(ctx):
            """Show bot statistics"""
            uptime = datetime.now() - self.stats['start_time']
            
            embed = discord.Embed(
                title="📊 Shinrai Bot Statistics",
                color=discord.Color.gold()
            )
            
            embed.add_field(name="Messages Processed", value=str(self.stats['messages_processed']))
            embed.add_field(name="Users Served", value=str(len(self.stats['users_served'])))
            embed.add_field(name="Uptime", value=str(uptime).split('.')[0])
            embed.add_field(name="Knowledge Base", value=str(len(self.shinrai.documents)))
            embed.add_field(name="Active Conversations", value=str(len(self.conversations)))
            embed.add_field(name="Model Version", value=self.shinrai.metadata.get('model_version', 'Unknown'))
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="info")
        async def show_info(ctx):
            """Show information about Shinrai"""
            embed = discord.Embed(
                title="🤖 About Shinrai AI",
                description="An uncensored AI chatbot that learns from websites",
                color=discord.Color.purple()
            )
            
            embed.add_field(name="Creator", value="Your Name")
            embed.add_field(name="Model Path", value=self.config['model_path'])
            embed.add_field(name="Documents", value=str(len(self.shinrai.documents)))
            embed.add_field(name="Last Trained", value=self.shinrai.metadata.get('last_trained', 'Unknown'))
            embed.add_field(name="Device", value=str(self.shinrai.device))
            embed.add_field(name="Features", value="Web scraping, Knowledge graph, Conversation memory")
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="summarize")
        async def summarize_knowledge(ctx):
            """Ask the bot to summarise its knowledge base"""
            summary = self.shinrai.response_generator._summarize_knowledge(self.shinrai.knowledge_graph)
            await ctx.send(summary)

        @self.bot.command(name="status")
        async def check_status(ctx):
            """Check bot status"""
            embed = discord.Embed(
                title="✅ Bot Status",
                color=discord.Color.green()
            )
            
            embed.add_field(name="Online", value="Yes")
            embed.add_field(name="Latency", value=f"{round(self.bot.latency * 1000)}ms")
            embed.add_field(name="Knowledge Base", value=str(len(self.shinrai.documents)))
            embed.add_field(name="Model Loaded", value="Yes" if self.shinrai.documents else "No")
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="memory")
        async def show_memory(ctx):
            """Show your conversation memory"""
            user_id = str(ctx.author.id)
            
            if user_id not in self.conversations or not self.conversations[user_id]:
                await ctx.send("📭 No conversation history found.")
                return
            
            history = self.conversations[user_id][-5:]  # Last 5 messages
            
            embed = discord.Embed(
                title="🧠 Your Recent Conversation",
                color=discord.Color.blue()
            )
            
            for i, msg in enumerate(history, 1):
                embed.add_field(
                    name=f"Message {i}",
                    value=f"**You:** {msg['user'][:50]}...\n**Bot:** {msg['bot'][:50]}...",
                    inline=False
                )
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="train")
        @commands.has_permissions(administrator=True)
        async def train_command(ctx, url: str, pages: int = 10):
            """Train on a website (Admin only) - Usage: !train <url> [pages]"""
            await ctx.send(f"🔄 Training on {url} (max {pages} pages). This may take a while...")
            
            async with ctx.typing():
                try:
                    # Run training in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    # let Shinrai handle scraping & training; passing args properly
                    await loop.run_in_executor(
                        None,
                        lambda: self.shinrai.train(url, source_type='web', max_pages=pages)
                    )

                    embed = discord.Embed(
                        title="✅ Training Complete",
                        color=discord.Color.green()
                    )
                    embed.add_field(name="URL", value=url)
                    embed.add_field(name="Pages", value=str(pages))
                    embed.add_field(name="Total Documents", value=str(len(self.shinrai.documents)))
                    
                    await ctx.send(embed=embed)
                except Exception as e:
                    await ctx.send(f"❌ Training failed: {str(e)}")
        
        @self.bot.command(name="config")
        @commands.has_permissions(administrator=True)
        async def show_config(ctx):
            """Show bot configuration (Admin only)"""
            embed = discord.Embed(
                title="⚙️ Bot Configuration",
                color=discord.Color.orange()
            )
            
            for key, value in self.config.items():
                if key != "token":  # Don't show token
                    embed.add_field(name=key, value=str(value), inline=False)
            
            await ctx.send(embed=embed)

        @self.bot.command(name="reload-config")
        @commands.has_permissions(administrator=True)
        async def reload_config(ctx):
            """Reload configuration from disk without restarting"""
            self.config = self.load_config()
            await ctx.send("🔄 Configuration reloaded.")
    
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
    
    async def get_ai_response(self, message: str, context) -> str:
        """Get response from Shinrai AI"""
        try:
            # Get conversation ID (user ID for DM, channel ID for guild)
            if isinstance(context, discord.Message):
                conv_id = str(context.author.id)
            else:
                conv_id = str(context.author.id)  # Use user ID for consistency
            
            # Get conversation history
            history = self.conversations.get(conv_id, [])
            
            # Get relevant documents
            loop = asyncio.get_event_loop()
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