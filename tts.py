# # tts.py

# def _escape_html(s: str) -> str:
#     return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# def _escape_js(s: str) -> str:
#     return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

# def inline_tts_button_inner(text: str, key: str) -> str:
#     """Return only the button + script (no container), for composing inline with text."""
#     safe_js = _escape_js(text)
#     return f"""
#     <button id="speak-btn-{key}"
#             onclick="toggleSpeak_{key}()"
#             style="background:none;border:none;cursor:pointer;font-size:20px">🔉</button>
#     <script>
#       (function(){{
#         let utterance_{key} = null;
#         let isSpeaking_{key} = false;
#         const btn = document.getElementById("speak-btn-{key}");
#         const isArabic = /[\\u0600-\\u06FF]/.test("{safe_js}");

#         function pickVoice(langStart) {{
#           const voices = window.speechSynthesis.getVoices() || [];
#           let v = voices.find(v => v.lang === langStart) ||
#                   voices.find(v => (v.lang || "").toLowerCase().startsWith(langStart.toLowerCase()));
#           return v || null;
#         }}

#         function buildUtterance() {{
#           const u = new SpeechSynthesisUtterance("{safe_js}");
#           if (isArabic) {{
#             const ar = pickVoice("ar");
#             if (ar) u.voice = ar;
#             u.lang = "ar-SA";
#           }} else {{
#             const en = pickVoice("en");
#             if (en) u.voice = en;
#             u.lang = "en-US";
#           }}
#           u.rate = 1.0;
#           u.pitch = 1.0;
#           u.onend = () => {{
#             isSpeaking_{key} = false;
#             if (btn) btn.textContent = "🔉";
#           }};
#           return u;
#         }}

#         function start() {{
#           window.speechSynthesis.cancel();
#           utterance_{key} = buildUtterance();
#           window.speechSynthesis.speak(utterance_{key});
#           isSpeaking_{key} = true;
#           btn.textContent = "🔇";
#         }}
#         function stop() {{
#           window.speechSynthesis.cancel();
#           isSpeaking_{key} = false;
#           btn.textContent = "🔉";
#         }}
#         window.toggleSpeak_{key} = function() {{
#           if (!isSpeaking_{key}) start(); else stop();
#         }};

#         if (speechSynthesis && speechSynthesis.onvoiceschanged !== undefined) {{
#           speechSynthesis.onvoiceschanged = function(){{}};
#         }}
#       }})();
#     </script>
#     """

# def response_with_tts_html(text: str, key: str) -> str:
#     """Return a single HTML block with the answer text and the 🔉 button inline."""
#     safe_html = _escape_html(text)
#     btn = inline_tts_button_inner(text, key=key)
#     return f"""
#     <div style="display:flex;align-items:flex-start;gap:10px;">
#       <div style="white-space:pre-wrap;flex:1;">{safe_html}</div>
#       {btn}
#     </div>
#     """

# def estimate_height(text: str) -> int:
#     """Heuristic height so the component fits in its bubble."""
#     # ~90 chars per line, ~22px per line
#     lines = max(1, len(text) // 90 + 1)
#     return min(240, max(80, 40 + lines * 22))
