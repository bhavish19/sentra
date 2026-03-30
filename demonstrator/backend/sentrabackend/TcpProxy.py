import asyncio
import threading
import grpc
import grpc.aio as grpc_aio
from .grpc import TCPProxyServiceServicer
from .grpc.generated import SentraBackend_GRPC_Services_pb2 as SentraBackend_GRPC_Services_pb2
from .grpc.generated import SentraBackend_GRPC_Services_pb2_grpc as SentraBackend_GRPC_Services_pb2_grpc
from .Log import log as log


class TCPProxyServicer(TCPProxyServiceServicer):
    """
    Registered alongside NodeMessageServiceServicer on the existing gRPC server.
    Each Stream() call represents one TCP connection on the remote bridge side.

    The servicer runs inside the gRPC event loop (the one created in
    startGRPCServer). It must NOT use asyncio.run_coroutine_threadsafe because
    it is already running inside that loop — just use plain await/yield.
    """

    def __init__(self, target_host: str, target_port: int):
        self.target_host = target_host
        self.target_port = target_port

    async def Stream(self, request_iterator, context):
        peer = context.peer()
        log(f"[TCPProxy] gRPC stream opened from {peer}")

        try:
            target_reader, target_writer = await asyncio.open_connection(
                self.target_host, self.target_port
            )
        except OSError as exc:
            log(
                f"[TCPProxy] Cannot connect to target "
                f"{self.target_host}:{self.target_port}: {exc}"
            )
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Target unreachable")
            return

        log(
            f"[TCPProxy] Connected to target {self.target_host}:{self.target_port}"
        )

        grpc_done = asyncio.Event()

        async def grpc_to_tcp():
            """Read PythonMsg.msg from the gRPC stream, write raw bytes to TCP."""
            try:
                async for message in request_iterator:
                    log(f"[TCPProxy] gRPC→TCP: {len(message.msg)} bytes")
                    async for message in request_iterator:
                        if message.HasField('python_msg'):
                            target_writer.write(message.python_msg.msg)
                            await target_writer.drain()
            except grpc.aio.AioRpcError as exc:
                log(f"[TCPProxy] gRPC read error: {exc}")
            finally:
                grpc_done.set()
                target_writer.close()

        async def tcp_to_grpc():
            """Read raw bytes from the TCP target, yield back as PythonMsg."""
            try:
                while not grpc_done.is_set():
                    data = await target_reader.read(4096)
                    if not data:
                        break
                    log(f"[TCPProxy] TCP→gRPC: {len(data)} bytes")
                    yield SentraBackend_GRPC_Services_pb2.ServerMessage(python_msg=SentraBackend_GRPC_Services_pb2.PythonMsg(msg=data))
            except OSError as exc:
                log(f"[TCPProxy] TCP read error: {exc}")

        pump = asyncio.create_task(grpc_to_tcp())
        async for msg in tcp_to_grpc():
            yield msg
        await pump

        log(f"[TCPProxy] gRPC stream closed from {peer}")


# ---------------------------------------------------------------------------
# TCP bridge (client side): listens for local TCP clients, tunnels via gRPC.
# Runs in its OWN thread + event loop, mirroring createGRPCServer's pattern,
# so it doesn't interfere with the gRPC server's loop.
# ---------------------------------------------------------------------------

class TCPProxyBridge:
    """
    Listens for plain TCP connections on listen_host:listen_port.
    Each accepted connection opens one bidirectional gRPC Stream() call to
    grpc_host:grpc_port and shuttles bytes in both directions.

    Lifecycle mirrors createGRPCServer:
        bridge = TCPProxyBridge(...)
        bridge.createBridge()   # starts background thread, returns immediately
    """

    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        grpc_host: str,
        grpc_port: int,
    ):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def createBridge(self):
        """
        Start the TCP bridge in a daemon thread (mirrors createGRPCServer)."""

        thread = threading.Thread(target=self._run_in_thread, daemon=True)
        thread.start()

    def stop(self):
        """Signal the bridge to shut down (can be called from any thread)."""
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._server.close)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_in_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()

    async def _run(self):
        self._server = await asyncio.start_server(
            self._handle_tcp_client,
            self.listen_host,
            self.listen_port,
        )
        addr = self._server.sockets[0].getsockname()
        log(f"[TCPBridge] Listening for TCP clients on {addr}")
        async with self._server:
            await self._server.serve_forever()

    async def _handle_tcp_client(
        self,
        tcp_reader: asyncio.StreamReader,
        tcp_writer: asyncio.StreamWriter,
    ):
        peer = tcp_writer.get_extra_info("peername")
        log(f"[TCPBridge] TCP client connected from {peer}")

        channel = grpc_aio.insecure_channel(f"{self.grpc_host}:{self.grpc_port}")
        stub = SentraBackend_GRPC_Services_pb2.TCPProxyServiceStub(channel)
        tcp_done = asyncio.Event()

        async def tcp_to_grpc_generator():
            """Yield raw TCP bytes wrapped in PythonMsg to the gRPC stream."""
            try:
                while not tcp_done.is_set():
                    data = await tcp_reader.read(4096)
                    if not data:
                        break
                    log(f"[TCPBridge] TCP→gRPC: {len(data)} bytes")
                    yield SentraBackend_GRPC_Services_pb2.NodeMessage(python_msg=SentraBackend_GRPC_Services_pb2.PythonMsg(msg=data))
            except OSError as exc:
                log(f"[TCPBridge] TCP read error: {exc}")
            finally:
                tcp_done.set()

        try:
            call = stub.Stream(tcp_to_grpc_generator())
            async for message in call:
                log(f"[TCPBridge] gRPC→TCP: {len(message.msg)} bytes")
                async for message in call:
                    if message.HasField('python_msg'):
                        tcp_writer.write(message.python_msg.msg)
                        await tcp_writer.drain()
        except grpc.aio.AioRpcError as exc:
            log(f"[TCPBridge] gRPC stream error: {exc}")
        finally:
            tcp_done.set()
            tcp_writer.close()
            await channel.close()
            log(f"[TCPBridge] TCP client disconnected: {peer}")
