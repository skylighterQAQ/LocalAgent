<template>
  <div class="app" :class="{ 'dark': isDark }">
    <div class="sidebar">
      <div class="sidebar-header">
        <h1>🤖 LocalAgent</h1>
        <button class="icon-btn" @click="toggleDark">{{ isDark ? '☀️' : '🌙' }}</button>
      </div>

      <div class="sidebar-section">
        <div class="section-title">Provider</div>
        <select v-model="selectedProvider" class="select" @change="onProviderChange">
          <option value="">Auto</option>
          <option value="ollama" :disabled="!providerStatus.ollama?.connected">
            Ollama{{ providerStatus.ollama?.connected ? '' : ' (offline)' }}
          </option>
          <option value="openai" :disabled="!providerStatus.openai?.connected">
            OpenAI{{ providerStatus.openai?.connected ? '' : ' (no key)' }}
          </option>
          <option value="wanqing" :disabled="!providerStatus.wanqing?.connected">
            Wanqing{{ providerStatus.wanqing?.connected ? '' : ' (no key)' }}
          </option>
          <option value="claude_code" :disabled="!providerStatus.claude_code?.connected">
            Claude Code{{ providerStatus.claude_code?.connected ? '' : ' (no key)' }}
          </option>
        </select>
      </div>

      <div class="sidebar-section">
        <div class="section-title">Model</div>
        <select v-model="selectedModel" class="select">
          <option v-for="m in availableModels" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>

      <div class="sidebar-section">
        <div class="section-title">Skill</div>
        <select v-model="selectedSkill" class="select">
          <option value="">Auto (All Tools)</option>
          <option v-for="s in skills" :key="s.name" :value="s.name">{{ s.name }}</option>
        </select>
        <div v-if="currentSkillDesc" class="skill-desc">{{ currentSkillDesc }}</div>
      </div>

      <div class="sidebar-section">
        <div class="section-title">Sessions</div>
        <button class="btn btn-sm" @click="newSession">+ New Session</button>
        <div class="session-list">
          <div
            v-for="s in sessions"
            :key="s.id"
            class="session-item"
            :class="{ active: s.id === currentSessionId }"
            @click="switchSession(s.id)"
          >
            <span>{{ s.preview || 'New Chat' }}</span>
          </div>
        </div>
      </div>

      <div class="sidebar-footer">
        <div :class="['status-dot', activeProviderConnected ? 'online' : 'offline']"></div>
        <span class="status-text">{{ activeProviderLabel }}</span>
      </div>
    </div>

    <div class="main">
      <ChatWindow
        :messages="currentMessages"
        :is-loading="isLoading"
        :session-id="currentSessionId"
        :model="selectedModel"
        :skill="selectedSkill"
        @send="handleSend"
        @clear="clearSession"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import ChatWindow from './components/ChatWindow.vue'
import { useChatStore } from './stores/chat'
import axios from 'axios'

const chatStore = useChatStore()
const isDark = ref(true)
const skills = ref<any[]>([])
const selectedProvider = ref('')
const selectedModel = ref('qwen2.5:7b')
const selectedSkill = ref('')

// Provider status from backend
const providerStatus = ref<Record<string, { connected: boolean; models?: string[]; default_model?: string }>>({})
// Models grouped by provider
const providerModels = ref<Record<string, string[]>>({})

const currentSessionId = computed(() => chatStore.currentSessionId)
const currentMessages = computed(() => chatStore.currentMessages)
const sessions = computed(() => chatStore.sessions)
const isLoading = computed(() => chatStore.isLoading)

const currentSkillDesc = computed(() => {
  const skill = skills.value.find(s => s.name === selectedSkill.value)
  return skill?.description || ''
})

// Models shown depend on which provider is selected
const availableModels = computed(() => {
  const p = selectedProvider.value
  if (!p) {
    // Auto: merge all models
    const all: string[] = []
    for (const models of Object.values(providerModels.value)) {
      for (const m of models) {
        if (!all.includes(m)) all.push(m)
      }
    }
    return all.length ? all : ['qwen2.5:7b']
  }
  return providerModels.value[p] || []
})

// Connection status of the currently selected provider (or any for "Auto")
const activeProviderConnected = computed(() => {
  const p = selectedProvider.value
  if (!p) {
    return Object.values(providerStatus.value).some(s => s.connected)
  }
  return providerStatus.value[p]?.connected ?? false
})

const activeProviderLabel = computed(() => {
  const p = selectedProvider.value
  const name = p ? p.replace('_', ' ') : 'Auto'
  return activeProviderConnected.value ? `${name} Connected` : `${name} Offline`
})

const toggleDark = () => { isDark.value = !isDark.value }
const newSession = () => { chatStore.newSession() }
const switchSession = (id: string) => { chatStore.switchSession(id) }
const clearSession = () => { chatStore.clearCurrentSession() }

function onProviderChange() {
  // When provider changes, reset model to the provider's default
  const p = selectedProvider.value
  if (p && providerStatus.value[p]?.default_model) {
    selectedModel.value = providerStatus.value[p].default_model!
  } else if (availableModels.value.length) {
    selectedModel.value = availableModels.value[0]
  }
}

const handleSend = async (message: string) => {
  await chatStore.sendMessage(
    message,
    selectedModel.value,
    selectedSkill.value || undefined,
    selectedProvider.value || undefined
  )
}

async function loadProviderStatus() {
  try {
    const res = await axios.get('/api/models/')
    const data = res.data
    const status: Record<string, any> = {}
    const models: Record<string, string[]> = {}

    for (const [name, info] of Object.entries<any>(data.providers || {})) {
      status[name] = { connected: info.available, default_model: info.default_model }
      models[name] = info.models || []
    }

    providerStatus.value = status
    providerModels.value = models

    // Set initial model if none selected yet
    if (!selectedModel.value || selectedModel.value === 'qwen2.5:7b') {
      const currentProvider = data.current_provider || ''
      if (currentProvider && models[currentProvider]?.length) {
        selectedModel.value = status[currentProvider]?.default_model || models[currentProvider][0]
      } else {
        // Pick first connected provider's default model
        for (const [name, info] of Object.entries(status)) {
          if (info.connected && models[name]?.length) {
            selectedModel.value = info.default_model || models[name][0]
            break
          }
        }
      }
    }
  } catch {
    // Silent fail – status stays offline
  }
}

onMounted(async () => {
  await loadProviderStatus()

  // Load skills
  try {
    const res = await axios.get('/api/skills/')
    skills.value = res.data.skills || []
  } catch { }
})
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

.app {
  display: flex;
  height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #1a1a2e;
  color: #e0e0e0;
}

.app:not(.dark) {
  background: #f5f5f5;
  color: #1a1a1a;
}

.sidebar {
  width: 260px;
  min-width: 260px;
  background: #16213e;
  border-right: 1px solid #0f3460;
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow-y: auto;
}

.app:not(.dark) .sidebar { background: #fff; border-right-color: #e0e0e0; }

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid #0f3460;
}

.sidebar-header h1 { font-size: 16px; font-weight: 700; color: #4fc3f7; }

.app:not(.dark) .sidebar-header { border-bottom-color: #e0e0e0; }
.app:not(.dark) .sidebar-header h1 { color: #1976d2; }

.sidebar-section { padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); }
.section-title { font-size: 11px; font-weight: 600; text-transform: uppercase; color: #888; margin-bottom: 8px; letter-spacing: 0.5px; }

.select {
  width: 100%;
  padding: 6px 10px;
  background: #0f3460;
  color: #e0e0e0;
  border: 1px solid #1e5799;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
}

.app:not(.dark) .select { background: #f0f0f0; color: #1a1a1a; border-color: #ccc; }

.skill-desc { font-size: 11px; color: #888; margin-top: 6px; line-height: 1.4; }

.btn { padding: 6px 12px; border-radius: 6px; border: none; cursor: pointer; font-size: 13px; font-weight: 500; }
.btn-sm { font-size: 12px; padding: 4px 10px; }
.btn { background: #4fc3f7; color: #000; }
.btn:hover { background: #81d4fa; }

.session-list { margin-top: 8px; max-height: 200px; overflow-y: auto; }
.session-item {
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  color: #aaa;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 2px;
}
.session-item:hover, .session-item.active { background: rgba(79, 195, 247, 0.15); color: #4fc3f7; }

.sidebar-footer { margin-top: auto; padding: 12px 16px; display: flex; align-items: center; gap: 8px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.online { background: #4caf50; }
.status-dot.offline { background: #f44336; }
.status-text { font-size: 12px; color: #888; }

.icon-btn { background: none; border: none; cursor: pointer; font-size: 16px; }

.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
</style>
