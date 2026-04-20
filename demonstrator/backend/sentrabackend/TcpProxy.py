import asyncio
import logging

logger = logging.getLogger(__name__)

from .Log import log as log
from .grpc.generated import SentraBackend_GRPC_Services_pb2 as pb2

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

    m_iNextPort:int

    def __init__(self, base_port: int):
        self.base_port = base_port
        #self.m_iNextPort=base_port

        # node_id -> asyncio.StreamWriter for the currently connected TCP client
        self._tcp_writers: dict[str, asyncio.StreamWriter] = {}

        # node_id -> asyncio.AbstractServer (the listening server for that node)
        self._servers: dict[str, asyncio.AbstractServer] = {}

        # node_id -> send_queue (set at registration time)
        self._send_queues: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # Called from NodeMessageServiceServicer when a node disconnects
    # ------------------------------------------------------------------

    async def closePortForNode(self, committee_index: int) -> None:
            """
            Close the TCP listener and any active TCP client for this node.
            """
            server = self._servers.pop(committee_index, None)
            if server:
                server.close()
                await server.wait_closed()
                log(f"[TCPProxy] Closed TCP port {self.base_port + committee_index} for committee member {committee_index}")

            writer = self._tcp_writers.pop(committee_index, None)
            if writer:
                writer.close()

            self._send_queues.pop(committee_index, None)

    # ------------------------------------------------------------------
    # Called from NodeStream.recv_messages() when a python_msg arrives
    # from a node — forward the bytes to the TCP client on that port
    # ------------------------------------------------------------------

    async def forwardToTCPClient(self, committee_index: int, data: bytes) -> None:
        """
        Write bytes received from a node's gRPC stream to the TCP client
        currently connected on that node's port.
        """
        writer = self._tcp_writers.get(committee_index)
        if writer is None:
            logger.warning(
                f"[TCPProxy] No TCP client connected for committee index {committee_index}, "
                f"dropping {len(data)} bytes"
            )
            return
        try:
            writer.write(data)
            await writer.drain()
            logger.debug(f"[TCPProxy] node→TCP (committee {committee_index}): {len(data)} bytes")
        except OSError as exc:
            logger.warning(f"[TCPProxy] Write to TCP client for committee index {committee_index} failed: {exc}")
            self._tcp_writers.pop(committee_index, None)

    # ------------------------------------------------------------------
    # Internal: handle one TCP client connection on a node's port
    # ------------------------------------------------------------------

    async def _handle_tcp_client(
            self,
            committee_index: int,
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            peer = writer.get_extra_info("peername")
            log(f"[TCPProxy] TCP client {peer} connected for committee member {committee_index} "
                f"on port {self.base_port + committee_index}")

            # Only one TCP client per committee member at a time — close any previous one
            old_writer = self._tcp_writers.get(committee_index)
            if old_writer:
                log(f"[TCPProxy] Replacing existing TCP client for committee index {committee_index}")
                old_writer.close()
                try:
                    await old_writer.wait_closed()
                except:
                    pass

            self._tcp_writers[committee_index] = writer

            send_queue = self._send_queues.get(committee_index)
            if send_queue is None:
                log(f"[TCPProxy] No send_queue for committee index {committee_index}, closing TCP client")
                writer.close()
                await writer.wait_closed()
                return

            try:
                while True:
                    data = await reader.read(4096)
                    if not data:
                        break
                    log(f"[TCPProxy] TCP→node (committee {committee_index}): {len(data)} bytes")

                    # Wrap in ServerMessage/PythonMsg and place on the node's send_queue.
                    # NodeStream's send loop will pick this up and yield it to the node
                    # over the existing gRPC stream — no separate gRPC channel needed.
                    message = pb2.ServerMessage(
                        python_msg=pb2.PythonMsg(msg=data)
                    )
                    await send_queue.put(message)

            except OSError as exc:
                log(f"[TCPProxy] TCP read error for committee index {committee_index}: {exc}")
            except Exception as exc:
                log(f"[TCPProxy] Unexpected error for committee index {committee_index}: {exc}")
            finally:
                self._tcp_writers.pop(committee_index, None)
                writer.close()
                try:
                    await writer.wait_closed()
                except:
                    pass
                log(f"[TCPProxy] TCP client {peer} disconnected from committee member {committee_index}")