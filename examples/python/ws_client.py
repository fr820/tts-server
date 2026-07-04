"""Send incremental text over the ElevenLabs-style WebSocket, collect audio."""

import asyncio
import base64
import json

import websockets

URL = "ws://localhost:8000/v1/text-to-speech/default/stream-input"


async def main() -> None:
    audio = b""
    msg: dict = {}
    async with websockets.connect(URL) as ws:
        for part in ["Hello, ", "welcome to our realtime voice demo.", ""]:
            await ws.send(json.dumps({"text": part}))
        while True:
            msg = json.loads(await ws.recv())
            audio += base64.b64decode(msg["audio"])
            if msg["isFinal"]:
                break
    print(f"received {len(audio)} bytes of pcm from {msg['backend']}")


if __name__ == "__main__":
    asyncio.run(main())
