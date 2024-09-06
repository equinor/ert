import asyncio
import sys
import websockets

async def hello(uri):
    async with websockets.connect(uri) as websocket:
        while True:
            greeting = await websocket.recv()
            print(f"Received: {greeting}")

eid = sys.argv[1]
asyncio.run(hello(f'ws://127.0.0.1:8000/experiments/{eid}/events'))
