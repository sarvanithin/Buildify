import { useState, useRef, useEffect } from 'react'
import { FloorPlan, Constraints } from '../types/floorplan'
import { sendDesignChat, ChatMsg } from '../api/client'

interface Props {
  plan: FloorPlan
  constraints: Constraints
  onPlanUpdate: (plan: FloorPlan, updatedConstraints: Constraints) => void
}

const STARTERS = [
  'Make the kitchen larger',
  'Add a home office',
  'I want a 4th bedroom',
  'Switch to craftsman style',
  'Remove the formal dining room',
  'Make it single story, no garage',
]

interface Message extends ChatMsg {
  delta?: Record<string, unknown>
  isQuestion?: boolean
  isOffTopic?: boolean
  validationError?: string
}

export default function ChatPanel({ plan, constraints, onPlanUpdate }: Props) {
  const [msgs, setMsgs] = useState<Message[]>([])
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
    const userMsg: Message = { role: 'user', content }
    const newMsgs = [...msgs, userMsg]
    setMsgs(newMsgs)
    setLoading(true)

    try {
      const apiMsgs: ChatMsg[] = newMsgs.map(m => ({ role: m.role, content: m.content }))
      const result = await sendDesignChat(
        plan,
        constraints as unknown as Record<string, unknown>,
        apiMsgs
      )

      const assistantMsg: Message = {
        role: 'assistant',
        content: result.explanation,
        delta: result.delta,
        isQuestion: result.is_question,
        isOffTopic: result.is_off_topic,
        validationError: result.validation_error,
      }
      setMsgs(prev => [...prev, assistantMsg])

      if (result.updated_plan && !result.is_question) {
        onPlanUpdate(
          result.updated_plan,
          result.updated_constraints as unknown as Constraints
        )
      }
    } catch {
      setMsgs(prev => [...prev, {
        role: 'assistant',
        content: 'Could not reach the AI service. Check your connection and try again.',
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-wrap">
      <div className="chat-header">
        <span className="chat-title">Design Assistant</span>
        <span className="chat-badge">Claude</span>
      </div>

      <div className="chat-messages">
        {msgs.length === 0 && (
          <div className="chat-welcome">
            <div className="chat-welcome-icon">🏠</div>
            <p>Describe a change and I'll update the floor plan instantly.</p>
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
              {msg.role === 'assistant' ? (
                <AssistantBubble msg={msg} />
              ) : (
                msg.content
              )}
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
          placeholder="Describe a change… (Enter to send)"
          disabled={loading}
        />
        <button className="chat-send-btn" onClick={() => send()} disabled={loading || !input.trim()}>
          ↑
        </button>
      </div>
    </div>
  )
}

function AssistantBubble({ msg }: { msg: Message }) {
  const hasDelta = msg.delta && Object.keys(msg.delta).length > 0
  const hasError = !!msg.validationError

  return (
    <div>
      <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>

      {msg.isOffTopic && (
        <div className="chat-delta chat-delta-offtopic">
          <span className="chat-delta-label">Design only</span>
          <span>Ask me about rooms, sizes, layout, or style.</span>
        </div>
      )}

      {hasDelta && !hasError && !msg.isOffTopic && (
        <div className="chat-delta">
          <span className="chat-delta-label">Plan updated</span>
          {Object.entries(msg.delta!).map(([k, v]) => (
            <span key={k} className="chat-delta-chip">
              {k}: {String(v)}
            </span>
          ))}
        </div>
      )}

      {hasError && (
        <div className="chat-delta chat-delta-error">
          <span className="chat-delta-label">Could not apply</span>
          <span>{msg.validationError}</span>
        </div>
      )}
    </div>
  )
}
