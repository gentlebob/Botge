/** @format */

import type { OmitPartialGroupDMChannel, Message } from 'discord.js';

import { storeMessage, findSimilarWithContext } from '../../api/vector-store.ts';
import { scoreReplyOpportunity, generateReply } from '../../api/ollama.ts';
import { logError } from '../../utils/public/log-error.ts';
import { ollamaConfig } from '../../api/config/ollama-config.ts';

type BufferEntry = {
  readonly author: string;
  readonly content: string;
  readonly timestamp: string;
  readonly tokens: number;
};

// Map of channelId → array of message objects
const buffers = new Map<string, BufferEntry[]>();
const lastReplyTime = new Map<string, number>();

async function persistToVectorStore(
  message: OmitPartialGroupDMChannel<Message>,
  author: string
): Promise<void> {
  try {
    await storeMessage({
      id: message.id,
      author,
      content: message.content,
      channelId: message.channel.id,
      timestamp: message.createdAt
    });
  } catch (error) {
    logError(error, 'Vector store write failed');
  }
}

async function retrieveContext(
  channelId: string,
  recentHistory: string
): Promise<readonly string[]> {
  if (!ollamaConfig.enabled) throw new Error('Ollama is not enabled.');

  try {
    const { ragResults, ragWindowSize } = ollamaConfig.behavior;

    return await findSimilarWithContext(recentHistory, channelId, ragResults, ragWindowSize);
  } catch (error) {
    logError(error, 'Vector store query failed');
    return [];
  }
}

function formatEntry(author: string, content: string): string {
  return `${author}: ${content}`;
}

/**
 * Rough token count for a formatted chat line. Gemma tokenizes every digit
 * separately (Discord mentions/emotes are 18-digit IDs), and ordinary text is
 * ~4 chars per token, so this slightly overestimates rather than under.
 */
function estimateTokens(line: string): number {
  const digits = line.replace(/\D/g, '').length;

  return digits + Math.ceil((line.length - digits) / 4) + 1;
}

/**
 * Format a channel's chat history as a readable string for the prompt.
 *
 * @param lastN - Only include the most recent N messages
 */
function getFormattedHistory(channelId: string, lastN?: number): string {
  const buffer = buffers.get(channelId) ?? [];
  const entries = lastN === undefined ? buffer : buffer.slice(-lastN);

  return entries.map((msg: BufferEntry) => formatEntry(msg.author, msg.content)).join('\n');
}

/**
 * Add a message to a channel's buffer.
 *
 * The buffer only grows until it passes the high watermark, then is cut down to
 * the low watermark in one go. Between cuts the prompt prefix stays identical,
 * so Ollama reuses its KV cache and only evaluates the newly appended messages.
 */
function addMessage(channelId: string, authorName: string, content: string): void {
  if (!ollamaConfig.enabled) throw new Error('Ollama is not enabled.');

  const buffer = buffers.get(channelId) ?? [];

  buffer.push({
    author: authorName,
    content,
    timestamp: new Date().toISOString(),
    tokens: estimateTokens(formatEntry(authorName, content))
  });

  const { historyHighWatermarkTokens, historyLowWatermarkTokens } = ollamaConfig.behavior;
  let total = buffer.reduce((sum: number, msg: BufferEntry) => sum + msg.tokens, 0);

  if (total > historyHighWatermarkTokens) {
    let drop = 0;
    while (drop < buffer.length - 1 && total > historyLowWatermarkTokens) {
      total -= buffer[drop].tokens;
      drop++;
    }
    buffer.splice(0, drop);
    console.log(`✂️  Trimmed ${drop} messages from ${channelId} history (~${total} tokens kept)`);
  }
  buffers.set(channelId, buffer);
}

/**
 * How many messages are buffered for a channel.
 */
function getBufferSize(channelId: string): number {
  return buffers.get(channelId)?.length ?? 0;
}
function isOnCooldown(channelId: string): boolean {
  if (!ollamaConfig.enabled) throw new Error('Ollama is not enabled.');

  const last = lastReplyTime.get(channelId) ?? 0;
  return (Date.now() - last) / 1000 < ollamaConfig.behavior.cooldownSeconds;
}
function setLastReplyTime(channelId: string): void {
  lastReplyTime.set(channelId, Date.now());
}

export async function ollamaMessageCreateHandler(
  message: OmitPartialGroupDMChannel<Message>,
  clientUserId: string
): Promise<void> {
  if (!ollamaConfig.enabled) throw new Error('Ollama is not enabled.');

  if (!ollamaConfig.activeChatChannels.includes(message.channel.id)) return;

  const channelId = message.channel.id;
  const authorName = message.author.displayName;
  const content = message.content.trim();

  // console.log('got message: ' + channelId + ' ' + content);

  // 1. Add to in-memory rolling buffer (fast, synchronous)
  addMessage(channelId, authorName, content);

  // 2. Persist to vector store in the background (async, non-blocking)
  //    This ensures all messages are indexed even when the bot doesn't reply
  void persistToVectorStore(message, authorName);

  // 3. Need a few messages of context before chiming in
  if (getBufferSize(channelId) < 3) return;

  // 4. Check per-channel cooldown / direct mention
  const direct_mention = content.includes(`<@${clientUserId}>`);
  if (!direct_mention && isOnCooldown(channelId)) return;
  setLastReplyTime(channelId);

  // 5. Random gate — skip most messages to avoid being annoying
  if (!direct_mention && Math.random() > ollamaConfig.behavior.evaluationChance) return;

  // 6. Score the opportunity using the recent buffer
  const recentHistory = getFormattedHistory(channelId);
  const channelName = 'name' in message.channel ? message.channel.name : channelId;
  // skip scoring for direct mention
  let shouldReply: boolean = direct_mention;
  if (!direct_mention) {
    shouldReply = await (async (): Promise<boolean> => {
      console.log(`🎲 Evaluating reply in #${channelName}...`);
      const scoring = await scoreReplyOpportunity(recentHistory).catch((err: unknown) => {
        console.error('❌ Scoring failed:', err instanceof Error ? err.message : String(err));
        return null;
      });
      if (scoring === null) return false;

      console.log(
        `📊 Score: ${scoring.score}/10 | Reply: ${scoring.shouldReply} | ${scoring.reason}`
      );

      if (!scoring.shouldReply) return false;
      return scoring.score >= ollamaConfig.behavior.replyScoreThreshold;
    })();
  }
  if (!shouldReply) return;

  // 7. Retrieve relevant past messages from vector store
  //    Use the recent chat as the semantic query to find topically relevant history
  //    Only the latest messages: the embedding model has a small context window.
  const retrievedContext = await retrieveContext(
    channelId,
    getFormattedHistory(channelId, ollamaConfig.behavior.ragQueryMessages)
  );
  // if (retrievedContext.length > 0) {
  console.log(`🔍 Retrieved ${retrievedContext.length} relevant past messages`);
  // }

  // 8. Generate reply and show typing indicator concurrently.
  //    sendTyping() expires after 10s, so we pulse it every 8s until the
  //    model finishes. This way the user sees "Alex is typing..." the whole
  //    time without any extra wait after the reply is ready.
  let typingTimer: ReturnType<typeof setInterval> | undefined = undefined;

  async function startTyping(): Promise<void> {
    try {
      await message.channel.sendTyping();
      typingTimer = setInterval(async () => {
        try {
          await message.channel.sendTyping();
        } catch {
          /* ignore */
        }
      }, 8000);
    } catch {
      // Missing Send Typing permission — not fatal
    }
  }

  function stopTyping(): void {
    clearInterval(typingTimer);
  }

  let reply: string | undefined;
  try {
    const [, generatedReply] = await Promise.all([
      startTyping(),
      generateReply(recentHistory, retrievedContext)
    ]);
    reply = generatedReply;
  } catch (error) {
    logError(error, 'Reply generation failed:');

    stopTyping();
    return;
  }

  stopTyping();
  if (!reply) return;

  // 9. Send the message and persist it too
  try {
    await message.reply(reply);
    // console.log('OLLAMA: ' + reply);

    // Add own reply to the in-memory buffer
    addMessage(channelId, ollamaConfig.botName, reply);

    // And index own reply in the vector store
    void storeMessage({
      id: `bot_${Date.now()}`,
      author: ollamaConfig.botName,
      content: reply,
      channelId,
      timestamp: new Date()
    });

    console.log(`✅ Sent: "${reply}"`);
  } catch (error) {
    logError(error, 'Failed to send');
  }
}
