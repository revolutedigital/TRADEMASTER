"use client";

import { useState, useEffect, useRef } from "react";

interface ChatMessage {
  id: string;
  userId: string;
  username: string;
  content: string;
  timestamp: number;
}

export function TraderChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [unread, setUnread] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as ChatMessage;
        setMessages((prev) => [...prev.slice(-99), msg]);
        if (!isOpen) setUnread((prev) => prev + 1);
      } catch {}
    };

    return () => ws.close();
  }, [isOpen]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function sendMessage(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: "chat_message", content: input.trim() }));
    setInput("");
  }

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => { setIsOpen(!isOpen); setUnread(0); }}
        className="fixed bottom-6 right-6 w-12 h-12 bg-blue-600 hover:bg-blue-700 rounded-full flex items-center justify-center text-white shadow-lg z-40 transition-colors"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        {unread > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center">
            {unread}
          </span>
        )}
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div className="fixed bottom-20 right-6 w-80 h-96 bg-[#141922] border border-gray-700 rounded-xl shadow-2xl flex flex-col z-40">
          <div className="p-3 border-b border-gray-700 flex justify-between items-center">
            <h3 className="text-white font-semibold text-sm">Trader Chat</h3>
            <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {messages.length === 0 ? (
              <p className="text-gray-500 text-center text-sm mt-8">No messages yet. Start the conversation!</p>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className="text-sm">
                  <span className="text-blue-400 font-medium">{msg.username}: </span>
                  <span className="text-gray-300">{msg.content}</span>
                  <span className="text-gray-600 text-xs ml-1">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={sendMessage} className="p-3 border-t border-gray-700">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type a message..."
                className="flex-1 bg-[#1a1f2e] text-white text-sm rounded-lg px-3 py-2 border border-gray-600 focus:border-blue-500 outline-none"
                maxLength={500}
              />
              <button type="submit" className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors">
                Send
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
