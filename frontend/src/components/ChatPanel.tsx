import { useState, useRef, useEffect } from 'react'
import { FloorPlan } from '../types/floorplan'
import { sendChatMessage, ChatMsg } from '../api/client'

interface Props {
  plan: FloorPlan
  onPlanUpdate: (plan: FloorPlan) => void
}

const STARTERS = [
  'Make the kitchen larger',
  'Add a mudroom near the garage',
  'Improve bedroom privacy',
  'Which rooms lack natural light?',
  'Suggest a better master suite layout',
]

export default function ChatPanel({ plan, onPlanUpdate }: Props) {
  const [msgs, setMsgs] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [msgs])

  async function send(text?: string) {
    const content = (text ?? input).trim()
    if (!content) return
    setInput('')
    const userMsg: ChatMsg = { role: 'user', content }
    const newMsgs = [...msgs, userMsg]
    setMsgs(newMsgs)
    setLoading(true)
    try {
      const { reply, updated_plan } = await sendChatMessage(plan, newMsgs)
      setMsgs(prev => [...prev, { role: 'assistant', content: reply }])
      if (updated_plan) {
        onPlanUpdate({ ...updated_plan, id: plan.id })
      }
    } catch (e) {
      setMsgs(prev => [...prev, { role: 'assistant', content: '⚠️ Could not reach Ollama. Make sure it is running.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-wrap">
      <div className="chat-header">
        <span className="chat-title">AI Design Assistant</span>
        <span className="chat-badge">Powered by Llama 3.2</span>
      </div>

      <div className="chat-messages">
        {msgs.length === 0 && (
          <div className="chat-welcome">
            <div className="chat-welcome-icon">🏠</div>
            <p>Ask me anything about your floor plan. I can suggest improvements, explain trade-offs, or modify room sizes.</p>
            <div className="chat-starters">
              {STARTERS.map(s => (
                <button key={s} className="starter-btn" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}

        {msgs.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg-${msg.role}`}>
            <div className="chat-bubble">
              {msg.role === 'assistant'
                ? <AssistantContent content={msg.content} />
                : msg.content
              }
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-msg chat-msg-assistant">
            <div className="chat-bubble chat-thinking">
              <span className="dot-flash" /><span className="dot-flash" /><span className="dot-flash" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          placeholder="Ask about your design… (Enter to send)"
          disabled={loading}
        />
        <button className="chat-send-btn" onClick={() => send()} disabled={loading || !input.trim()}>
          ↑
        </button>
      </div>
    </div>
  )
}

/** Strip ```json...``` blocks from assistant replies for display */
function AssistantContent({ content }: { content: string }) {
  const cleaned = content.replace(/```json[\s\S]*?```/g, '*(floor plan updated)*').trim()
  return <span style={{ whiteSpace: 'pre-wrap' }}>{cleaned}</span>
}
