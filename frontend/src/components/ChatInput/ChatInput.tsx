import './ChatInput.css'

type Props = {
  input: string
  loading: boolean
  onChange: (value: string) => void
  onSubmit: (e: React.FormEvent) => void
}

export function ChatInput({ input, loading, onChange, onSubmit }: Props) {
  return (
    <form className="input-row" onSubmit={onSubmit}>
      <input
        value={input}
        onChange={e => onChange(e.target.value)}
        placeholder="Écris un message..."
        disabled={loading}
      />
      <button type="submit" disabled={loading || !input.trim()}>
        Envoyer
      </button>
    </form>
  )
}
