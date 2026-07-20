import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface Session {
  id: string
  messages: Message[]
  preview: string
}

export const useChatStore = defineStore('chat', () => {
  const sessions = ref<Session[]>([
    { id: generateId(), messages: [], preview: '' }
  ])
  const currentSessionId = ref(sessions.value[0].id)
  const isLoading = ref(false)

  const currentSession = computed(() =>
    sessions.value.find(s => s.id === currentSessionId.value) || sessions.value[0]
  )

  const currentMessages = computed(() => currentSession.value?.messages || [])

  function generateId() {
    return Math.random().toString(36).substr(2, 9)
  }

  function newSession() {
    const session: Session = { id: generateId(), messages: [], preview: '' }
    sessions.value.unshift(session)
    currentSessionId.value = session.id
  }

  function switchSession(id: string) {
    currentSessionId.value = id
  }

  function clearCurrentSession() {
    const session = currentSession.value
    if (session) {
      session.messages = []
      session.preview = ''
      // Also clear on server
      axios.delete(`/api/chat/${session.id}`).catch(() => {})
    }
  }

  async function sendMessage(
    message: string,
    model: string = 'qwen2.5:7b',
    skill?: string,
    provider?: string
  ) {
    const session = currentSession.value
    if (!session || isLoading.value) return

    // Add user message
    const userMsg: Message = { role: 'user', content: message, timestamp: new Date() }
    session.messages.push(userMsg)
    session.preview = message.slice(0, 40)

    // Add placeholder assistant message for streaming
    const assistantMsg: Message = { role: 'assistant', content: '', timestamp: new Date() }
    session.messages.push(assistantMsg)
    isLoading.value = true

    try {
      // Use WebSocket for streaming
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsHost = window.location.hostname === 'localhost' ? 'localhost:8080' : window.location.host
      const ws = new WebSocket(`${wsProtocol}//${wsHost}/api/chat/ws/${session.id}`)

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => {
          ws.send(JSON.stringify({ message, model, skill: skill || null, provider: provider || null }))
        }

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data)
          if (data.type === 'chunk') {
            assistantMsg.content += data.content
          } else if (data.type === 'done') {
            ws.close()
            resolve()
          } else if (data.type === 'error') {
            assistantMsg.content = `Error: ${data.message}`
            ws.close()
            reject(new Error(data.message))
          }
        }

        ws.onerror = () => {
          // Fallback to HTTP
          axios.post('/api/chat/', {
            message,
            session_id: session.id,
            model,
            skill: skill || null,
            provider: provider || null,
          }).then(res => {
            assistantMsg.content = res.data.response
            resolve()
          }).catch(err => {
            assistantMsg.content = `Error: ${err.message}`
            reject(err)
          })
        }

        ws.onclose = () => resolve()
      })
    } catch (error: any) {
      if (!assistantMsg.content) {
        assistantMsg.content = `Error: ${error.message || 'Unknown error'}`
      }
    } finally {
      isLoading.value = false
    }
  }

  return {
    sessions,
    currentSessionId,
    currentMessages,
    isLoading,
    newSession,
    switchSession,
    clearCurrentSession,
    sendMessage,
  }
})

function generateId() {
  return Math.random().toString(36).substr(2, 9)
}
