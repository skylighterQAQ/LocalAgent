<template>
  <div class="chat-window">
    <div class="messages-container" ref="messagesContainer">
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-icon">🤖</div>
        <h2>LocalAgent</h2>
        <p>Your local AI assistant powered by Ollama</p>
        <div class="suggestions">
          <div class="suggestion" v-for="s in suggestions" :key="s" @click="sendSuggestion(s)">
            {{ s }}
          </div>
        </div>
      </div>

      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        class="message"
        :class="msg.role"
      >
        <div class="message-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</div>
        <div class="message-content">
          <div class="message-header">
            <span class="message-role">{{ msg.role === 'user' ? 'You' : 'Assistant' }}</span>
            <span class="message-time">{{ formatTime(msg.timestamp) }}</span>
          </div>
          <div class="message-body" v-html="renderMarkdown(msg.content)"></div>
        </div>
      </div>

      <div v-if="isLoading" class="message assistant loading-msg">
        <div class="message-avatar">🤖</div>
        <div class="message-content">
          <div class="message-header"><span class="message-role">Assistant</span></div>
          <div class="message-body">
            <span class="typing-indicator">
              <span></span><span></span><span></span>
            </span>
          </div>
        </div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-toolbar">
        <button class="tool-btn" @click="$emit('clear')" title="Clear conversation">🗑</button>
        <span class="session-id-label">Session: {{ sessionId?.slice(0, 8) || 'none' }}</span>
      </div>
      <div class="input-row">
        <textarea
          ref="inputRef"
          v-model="inputText"
          class="message-input"
          placeholder="Ask me anything... (Shift+Enter for newline)"
          rows="1"
          @keydown="handleKeydown"
          @input="autoResize"
          :disabled="isLoading"
        ></textarea>
        <button
          class="send-btn"
          :disabled="!inputText.trim() || isLoading"
          @click="sendMessage"
        >
          {{ isLoading ? '⏳' : '➤' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import { marked } from 'marked'

const props = defineProps<{
  messages: Array<{ role: string; content: string; timestamp: Date }>
  isLoading: boolean
  sessionId: string | null
  model: string
  skill: string
}>()

const emit = defineEmits<{
  send: [message: string]
  clear: []
}>()

const inputText = ref('')
const inputRef = ref<HTMLTextAreaElement>()
const messagesContainer = ref<HTMLDivElement>()

const suggestions = [
  '📁 List files in the current directory',
  '🔍 Search the web for latest AI news',
  '🐍 Write a Python script to sort a list',
  '📊 Explain data analysis techniques',
  '💻 What is my system info?',
  '🌐 Fetch the content of example.com',
]

const renderMarkdown = (content: string): string => {
  if (!content) return ''
  try {
    return marked.parse(content) as string
  } catch {
    return content
  }
}

const formatTime = (ts: Date): string => {
  if (!ts) return ''
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const sendMessage = () => {
  const msg = inputText.value.trim()
  if (!msg || props.isLoading) return
  emit('send', msg)
  inputText.value = ''
  if (inputRef.value) {
    inputRef.value.style.height = 'auto'
  }
}

const sendSuggestion = (s: string) => {
  inputText.value = s
  sendMessage()
}

const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

const autoResize = () => {
  if (inputRef.value) {
    inputRef.value.style.height = 'auto'
    inputRef.value.style.height = Math.min(inputRef.value.scrollHeight, 200) + 'px'
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

watch(() => props.messages.length, scrollToBottom)
watch(() => props.isLoading, scrollToBottom)
</script>

<style scoped>
.chat-window {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #1a1a2e;
}

.app:not(.dark) .chat-window { background: #f5f5f5; }

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px 16px;
  scroll-behavior: smooth;
}

.empty-state {
  text-align: center;
  margin-top: 60px;
  color: #888;
}
.empty-icon { font-size: 48px; margin-bottom: 12px; }
.empty-state h2 { color: #4fc3f7; font-size: 24px; margin-bottom: 8px; }
.empty-state p { color: #888; margin-bottom: 24px; }

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  max-width: 600px;
  margin: 0 auto;
}
.suggestion {
  padding: 8px 14px;
  background: rgba(79, 195, 247, 0.1);
  border: 1px solid rgba(79, 195, 247, 0.3);
  border-radius: 20px;
  cursor: pointer;
  font-size: 13px;
  color: #4fc3f7;
  transition: all 0.2s;
}
.suggestion:hover { background: rgba(79, 195, 247, 0.2); }

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
}
.message.user { flex-direction: row-reverse; }

.message-avatar {
  font-size: 24px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.message-content {
  flex: 1;
  min-width: 0;
  max-width: calc(100% - 48px);
}

.message.user .message-content {
  background: #0f3460;
  border-radius: 12px 2px 12px 12px;
  padding: 10px 14px;
}

.message.assistant .message-content {
  background: rgba(255,255,255,0.04);
  border-radius: 2px 12px 12px 12px;
  padding: 10px 14px;
  border: 1px solid rgba(255,255,255,0.07);
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.message-role { font-size: 12px; font-weight: 600; color: #4fc3f7; }
.message-time { font-size: 11px; color: #666; }

.message-body {
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
}

.message-body :deep(pre) {
  background: #0d1117;
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
  margin: 8px 0;
}
.message-body :deep(code) {
  background: rgba(255,255,255,0.1);
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 13px;
}
.message-body :deep(pre code) { background: none; padding: 0; }
.message-body :deep(p) { margin-bottom: 8px; }
.message-body :deep(ul), .message-body :deep(ol) { padding-left: 20px; }

.typing-indicator {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  padding: 4px 0;
}
.typing-indicator span {
  width: 8px; height: 8px;
  background: #4fc3f7;
  border-radius: 50%;
  animation: bounce 1.2s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
  40% { transform: scale(1); opacity: 1; }
}

.input-area {
  padding: 12px 16px;
  background: #16213e;
  border-top: 1px solid rgba(255,255,255,0.07);
}
.input-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.tool-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 16px;
  padding: 2px 6px;
  border-radius: 4px;
  opacity: 0.6;
  transition: opacity 0.2s;
}
.tool-btn:hover { opacity: 1; }
.session-id-label { font-size: 11px; color: #555; margin-left: auto; }

.input-row {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.message-input {
  flex: 1;
  padding: 10px 14px;
  background: #0f3460;
  color: #e0e0e0;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px;
  font-size: 14px;
  resize: none;
  max-height: 200px;
  line-height: 1.5;
  font-family: inherit;
  transition: border-color 0.2s;
}
.message-input:focus { outline: none; border-color: #4fc3f7; }
.message-input:disabled { opacity: 0.6; }

.send-btn {
  padding: 10px 16px;
  background: #4fc3f7;
  color: #000;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  font-size: 18px;
  font-weight: bold;
  transition: all 0.2s;
  min-width: 48px;
}
.send-btn:hover:not(:disabled) { background: #81d4fa; transform: scale(1.05); }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
