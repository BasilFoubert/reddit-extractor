import type { RefObject } from 'react'
import type { Message } from '@/types'
import './MessageList.css'

type Props = {
  messages: Message[]
  loading: boolean
  bottomRef: RefObject<HTMLDivElement | null>
}

export function MessageList({ messages, loading, bottomRef }: Props) {
  return (
    <div className="messages">
      {messages.map((msg, i) => (
        <div key={i} className={`message ${msg.role}`}>
          {msg.content}
        </div>
      ))}
      {loading && <div className="message assistant">...</div>}
      <div ref={bottomRef} />
    </div>
  )
}
