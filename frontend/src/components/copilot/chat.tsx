"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Bot, Sparkles, User } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/primitives";
import { api, streamSSE } from "@/lib/api";
import { useSettings } from "@/lib/store";
import type { ChatAction, ChatAnswer, ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";

export function CopilotChat({
  studyId,
  onAction,
  className,
}: {
  studyId: string;
  onAction: (a: ChatAction) => void;
  className?: string;
}) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [suggestions, setSuggestions] = React.useState<string[]>([]);
  const [input, setInput] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const scroller = React.useRef<HTMLDivElement>(null);
  const delay = useSettings((s) => s.chatDelay);

  React.useEffect(() => {
    api
      .chatHistory(studyId)
      .then((h) => {
        setMessages(
          h.messages.map((m) => ({ role: m.role as "user" | "assistant", content: m.content, intent: m.intent, actions: m.actions, suggestions: m.suggestions })),
        );
        setSuggestions(h.suggestions);
      })
      .catch(() => undefined);
  }, [studyId]);

  React.useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", content: q }, { role: "assistant", content: "", streaming: true }]);
    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));
    try {
      await streamSSE(`/api/copilot/${studyId}/chat`, { message: q, stream: true, delay_ms: delay }, (event, data) => {
        if (event === "token") patch((m) => ({ ...m, content: m.content + String(data) }));
        else if (event === "meta") patch((m) => ({ ...m, intent: (data as { intent: string }).intent }));
        else if (event === "done") {
          const a = data as ChatAnswer;
          patch((m) => ({ ...m, content: a.text, actions: a.actions, streaming: false, intent: a.intent }));
          setSuggestions(a.suggestions);
        }
      });
    } catch {
      patch((m) => ({ ...m, content: "The Copilot engine is not reachable — is the backend running?", streaming: false }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={cn("flex h-full min-h-0 flex-col", className)}>
      <div ref={scroller} className="scrollbar-thin min-h-0 flex-1 space-y-3 overflow-y-auto px-1 py-2" aria-live="polite">
        {!messages.length && (
          <div className="rounded-2xl border border-border bg-surface-2 p-4 text-sm text-muted">
            <div className="mb-1.5 flex items-center gap-2 font-medium text-fg">
              <Sparkles className="size-4 text-cyan" /> MediVision Copilot
            </div>
            A self-built clinical language engine (intent model + knowledge-base retrieval + rule-based composers). Answers are
            grounded in this study&apos;s outputs. No external AI.
          </div>
        )}
        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className={cn("flex gap-2.5", m.role === "user" && "flex-row-reverse")}>
              <div
                className={cn(
                  "mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg border",
                  m.role === "user" ? "border-border bg-surface-2 text-muted" : "border-cyan/30 bg-cyan/10 text-cyan",
                )}
              >
                {m.role === "user" ? <User className="size-3.5" /> : <Bot className="size-3.5" />}
              </div>
              <div className={cn("max-w-[86%]", m.role === "user" && "text-right")}>
                <div
                  className={cn(
                    "inline-block whitespace-pre-wrap rounded-2xl px-3.5 py-2.5 text-left text-[13px] leading-relaxed",
                    m.role === "user" ? "bg-accent-gradient text-white" : "border border-border bg-surface-2 text-fg",
                  )}
                  data-testid={m.role === "assistant" ? "assistant-message" : undefined}
                >
                  {m.content}
                  {m.streaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse rounded-sm bg-cyan align-middle" />}
                </div>
                {m.role === "assistant" && !m.streaming && (m.actions?.length || m.intent) ? (
                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                    {m.actions?.map((a, j) => (
                      <button
                        key={j}
                        onClick={() => onAction(a)}
                        className="rounded-full border border-cyan/30 bg-cyan/10 px-2.5 py-1 text-[11px] font-medium text-cyan transition hover:bg-cyan/20"
                        data-testid="action-chip"
                      >
                        {a.label}
                      </button>
                    ))}
                    {m.intent && <span className="num text-[10px] text-subtle">intent: {m.intent}</span>}
                  </div>
                ) : null}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
      <div className="flex flex-wrap gap-1.5 py-2">
        {suggestions.map((s) => (
          <button
            key={s}
            disabled={busy}
            onClick={() => send(s)}
            className="rounded-full border border-border bg-surface-2 px-2.5 py-1 text-[11px] text-muted transition hover:border-cyan/40 hover:text-fg disabled:opacity-50"
            data-testid="suggestion-chip"
          >
            {s}
          </button>
        ))}
      </div>
      <form
        className="flex items-end gap-2 rounded-2xl border border-border bg-surface-2 p-1.5 focus-within:border-cyan/40"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          rows={1}
          placeholder="Ask about this study… e.g. “why effusion?”"
          className="max-h-28 min-h-9 flex-1 resize-none bg-transparent px-2.5 py-2 text-sm text-fg placeholder:text-subtle focus:outline-none"
          aria-label="Ask the Copilot"
          data-testid="chat-input"
        />
        <Button type="submit" variant="primary" size="icon" disabled={busy || !input.trim()} aria-label="Send">
          <ArrowUp />
        </Button>
      </form>
    </div>
  );
}
