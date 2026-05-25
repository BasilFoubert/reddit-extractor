import axios from 'axios'

const API = 'http://localhost:8000'

export async function createConversation(): Promise<{ threadId: string; initialMessage: string }> {
  const res = await axios.post(`${API}/conversations`)
  return { threadId: res.data.thread_id, initialMessage: res.data.initial_message }
}

export async function sendMessage(threadId: string, content: string): Promise<string> {
  const res = await axios.post(`${API}/conversations/${threadId}/messages`, { content })
  return res.data.content
}
