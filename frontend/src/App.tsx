import './App.css'
import { ChatInput } from '@/components/ChatInput/ChatInput'
import { MessageList } from '@/components/MessageList/MessageList'
import { useChat } from '@/hooks/useChat'

export default function App() {
  const { messages, input, setInput, loading, handleSubmit, bottomRef } = useChat()

  return (
    <div className="chat">
      <MessageList messages={messages} loading={loading} bottomRef={bottomRef} />
      <ChatInput input={input} loading={loading} onChange={setInput} onSubmit={handleSubmit} />
    </div>
  )
}
