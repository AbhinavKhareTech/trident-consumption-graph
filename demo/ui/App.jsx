import { useState } from "react";

const DEMO_URL = "http://localhost:8000";

export default function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;
    setMessages((prev) => [...prev, { role: "user", text: input }]);
    setLoading(true);
    try {
      const res = await fetch(`${DEMO_URL}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.summary,
          intent: data.intent,
          language: data.language,
          entities: data.entities,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [...prev, { role: "error", text: String(e) }]);
    }
    setLoading(false);
    setInput("");
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">BGI Trident Demo</h1>
      <p className="text-sm text-gray-500 mb-4">
        Try: "Meghana's biryani order maaDu, Harpic nu add maaDu, Saturday ge
        table book maaDu"
      </p>
      <div className="border rounded-lg p-4 h-96 overflow-y-auto mb-4 bg-gray-50">
        {messages.map((m, i) => (
          <div key={i} className={`mb-3 ${m.role === "user" ? "text-right" : ""}`}>
            <span className={`inline-block px-3 py-2 rounded-lg ${
              m.role === "user" ? "bg-blue-600 text-white" : "bg-white border"
            }`}>
              {m.text}
            </span>
            {m.intent && (
              <div className="text-xs text-gray-400 mt-1">
                Intent: {m.intent} | Lang: {m.language}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          className="flex-1 border rounded-lg px-4 py-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Speak to Trident..."
        />
        <button
          className="bg-blue-600 text-white px-6 py-2 rounded-lg"
          onClick={handleSend}
          disabled={loading}
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
