// node >= 21 (built-in WebSocket). Streams text in, collects base64 audio out.
const url = "ws://localhost:8000/v1/text-to-speech/default/stream-input";
const ws = new WebSocket(url);
let bytes = 0;

ws.onopen = () => {
  for (const text of ["Hello, ", "welcome to our realtime voice demo.", ""]) {
    ws.send(JSON.stringify({ text }));
  }
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  bytes += Buffer.from(msg.audio, "base64").length;
  if (msg.isFinal) {
    console.log(`received ${bytes} bytes of pcm from ${msg.backend}`);
    ws.close();
  }
};
