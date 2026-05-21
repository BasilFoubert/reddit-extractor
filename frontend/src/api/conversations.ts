import axios from 'axios'

const API = 'http://localhost:8000'

export async function createConversation(): Promise<string> {
  const res = await axios.post(`${API}/conversations`)
  return res.data.thread_id
}

export async function sendMessage(threadId: string, content: string): Promise<string> {
  const res = await axios.post(`${API}/conversations/${threadId}/messages`, { content })
  return res.data.content
}
