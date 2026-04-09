import asyncio
import logging

logger = logging.getLogger(__name__)


class NodeTCPProxy:
    """
    Manages one TCP listener per registered node on base_port + node_index.

    Data flow:
        TCP client → port (base_port + node_index)
            → wrapped in PythonMsg → placed on node's send_queue → gRPC → node

        node → gRPC → python_msg received in NodeStream.recv_messages()
            → forwardToTCPClient(node_id, data)
            → written to the TCP client connected on that node's port
    """

    def __init__(self, base_port: int):
        self.base_port = base_port

        # node_id -> asyncio.StreamWriter for the currently connected TCP client
        self._tcp_writers: dict[str, asyncio.StreamWriter] = {}

        # node_id -> asyncio.AbstractServer (the listening server for that port)
        self._servers: dict[str, asyncio.AbstractServer] = {}

        # node_id -> send_queue (set at registration time)
        self._send_queues: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # Called from NodeMessageServiceServicer when a node registers
    # ------------------------------------------------------------------

    async def openPortForNode(
        self,
        node_id: int,
        send_queue: asyncio.Queue,
    ) -> int:
        """
        Open a TCP listener on base_port + node_index for the given node.
        Returns the port number that was opened.
        Already called from within the gRPC event loop, so plain await is fine.
        """
        if node_id in self._servers:
            logger.warning(f"[TCPProxy] Port already open for node {node_id}, skipping")
            return self.base_port + node_id

        port = self.base_port + node_id
        self._send_queues[node_id] = send_queue

        server = await asyncio.start_server(
            lambda r, w: self._handle_tcp_client(node_id, r, w),
            host="0.0.0.0",
            port=port,
        )
        self._servers[node_id] = server
        asyncio.create_task(server.serve_forever())

        logger.info(f"[TCPProxy] Opened TCP port {port} for node {node_id}")
        return port

    # ------------------------------------------------------------------
    # Called from NodeMessageServiceServicer when a node disconnects
    # ------------------------------------------------------------------

    async def closePortForNode(self, node_id: str) -> None:
        """
        Close the TCP listener and any active TCP client for this node.
        """
        server = self._servers.pop(node_id, None)
        if server:
            server.close()
            await server.wait_closed()
            logger.info(f"[TCPProxy] Closed TCP port for node {node_id}")

        writer = self._tcp_writers.pop(node_id, None)
        if writer:
            writer.close()

        self._send_queues.pop(node_id, None)

    # ------------------------------------------------------------------
    # Called from NodeStream.recv_messages() when a python_msg arrives
    # from a node — forward the bytes to the TCP client on that port
    # ------------------------------------------------------------------

    async def forwardToTCPClient(self, node_id: str, data: bytes) -> None:
        """
        Write bytes received from a node's gRPC stream to the TCP client
        currently connected on that node's port.
        """
        writer = self._tcp_writers.get(node_id)
        if writer is None:
            logger.warning(
                f"[TCPProxy] No TCP client connected for node {node_id}, dropping {len(data)} bytes"
            )
            return
        try:
            writer.write(data)
            await writer.drain()
            logger.debug(f"[TCPProxy] node→TCP ({node_id}): {len(data)} bytes")
        except OSError as exc:
            logger.warning(f"[TCPProxy] Write to TCP client for node {node_id} failed: {exc}")
            self._tcp_writers.pop(node_id, None)

    # ------------------------------------------------------------------
    # Internal: handle one TCP client connection on a node's port
    # ------------------------------------------------------------------

    async def _handle_tcp_client(
        self,
        node_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.info(f"[TCPProxy] TCP client {peer} connected for node {node_id}")

        # Only one TCP client per node at a time — close any previous one
        old_writer = self._tcp_writers.get(node_id)
        if old_writer:
            logger.warning(
                f"[TCPProxy] Replacing existing TCP client for node {node_id}"
            )
            old_writer.close()

        self._tcp_writers[node_id] = writer

        send_queue = self._send_queues.get(node_id)
        if send_queue is None:
            logger.error(f"[TCPProxy] No send_queue for node {node_id}, closing TCP client")
            writer.close()
            return

        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                logger.debug(f"[TCPProxy] TCP→node ({node_id}): {len(data)} bytes")

                # Wrap in ServerMessage/PythonMsg and place on the node's send_queue.
                # NodeStream's send loop will pick this up and yield it to the node
                # over the existing gRPC stream — no separate gRPC channel needed.
                from .generated import SentraBackend_GRPC_Services_pb2 as pb2
                message = pb2.ServerMessage(
                    python_msg=pb2.PythonMsg(msg=data)
                )
                await send_queue.put(message)

        except OSError as exc:
            logger.warning(f"[TCPProxy] TCP read error for node {node_id}: {exc}")
        finally:
            self._tcp_writers.pop(node_id, None)
            writer.close()
            logger.info(f"[TCPProxy] TCP client {peer} disconnected from node {node_id}")