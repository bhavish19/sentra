import asyncio
import time

from .generated import NodeMessageServiceServicer as _NodeMessageServiceServicer
from .generated import SentraBackend_GRPC_Services_pb2 as SentraBackend_GRPC_Services_pb2

from ..SentraNode import SentraNode
from ..SentraNodeAttributeGenerator import SentraNodeAttributeGenerator
from ..SentraNodeList import SentraNodeList
from ..ComitteeSelection import CommitteeSelection
from ..Attestation import Attestation
from ..CommandLineOptions import CommandLineOptions
from ..Log import log as log
from ..TcpProxy import NodeTCPProxy

class NodeMessageServiceServicer(_NodeMessageServiceServicer):

    m_nodeGenerator:SentraNodeAttributeGenerator
    m_nodeList:SentraNodeList
    m_committee:SentraNodeList
    m_sortedCommittee: list[SentraNode]|None
    m_GRPC_Loop:asyncio.AbstractEventLoop
    m_commandLineOptions:CommandLineOptions
    m_bCommitteeSelected:bool

    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeList:SentraNodeList,commandlineOptions:CommandLineOptions):
        self.m_nodeGenerator=nodeGenerator
        self.m_nodeList=nodeList
        self.m_GRPC_Loop=asyncio.get_event_loop()
        self.m_commandLineOptions=commandlineOptions
        self.m_bCommitteeSelected=False
        self.m_sortedCommittee = None
        self._servers: dict[int, asyncio.Server] = {}
        self._send_queues: dict[int, asyncio.Queue] = {}
        self.base_port = commandlineOptions.getTcpProxyBasePort()

        self.m_tcpProxy = NodeTCPProxy(self.base_port)
        #self._node_index: dict[str, int] = {}
        self.m_currentCommittee: SentraNodeList | None = None

    def sendMessageToNode(self,node_id:str,message:SentraBackend_GRPC_Services_pb2.ServerMessage)->None:
        sendQueue: asyncio.Queue[object]|None=self.m_nodeList.getSendQueue(node_id)
        if(not sendQueue is None):
            asyncio.run_coroutine_threadsafe(sendQueue.put(message),self.m_GRPC_Loop)

    def putMessageInNodeSendQueue(self,sendQueue: asyncio.Queue[object],message:SentraBackend_GRPC_Services_pb2.ServerMessage)->None:
        asyncio.run_coroutine_threadsafe(sendQueue.put(message),self.m_GRPC_Loop)

    def registerNode(self, message:object)->tuple[object,str|None,asyncio.Queue[object]|None]:
        if not message.HasField('register'):
            return (None,None,None)
        register_message:object=message.register
        node_id:str|None = register_message.node_id
        node_grpc_url:str|None=register_message.node_grpc_url

        if not node_id or not isinstance(node_id,str):
            return (SentraBackend_GRPC_Services_pb2.RegisterResponse(
                success=False,
                message="Node ID cannot be empty"
            ),None,None)
        node:SentraNode=self.m_nodeGenerator.generateNode(node_id)
        node.m_strInterNodeCommunicationGRPC_URL=node_grpc_url
        if(self.m_nodeList.add(node)):
            log(f"Node registered: {node_id} - Node list now has {self.m_nodeList.len()} entries")
            return (SentraBackend_GRPC_Services_pb2.RegisterResponse(
                success=True,
                message=f"Node {node_id} registered successfully"
            ),node_id,node.getSendQueue())
        else:
            log(f"Node already registered: {node_id} - Will not add this connection")
            return (SentraBackend_GRPC_Services_pb2.RegisterResponse(
                success=False,
                message=f"Node {node_id} already registered"
            ),None,None)

    def generatePythonMessage(self, msg)->SentraBackend_GRPC_Services_pb2.PythonMsg:
        return SentraBackend_GRPC_Services_pb2.PythonMsg(msg=msg)


    def generateAttestationRequest(self)->SentraBackend_GRPC_Services_pb2.AttestationRequest:
        return SentraBackend_GRPC_Services_pb2.AttestationRequest(nonce="Nonce")

    async def requestCommitteeJoin(self,committee:SentraNodeList):
        log("Send requests to all Nodes of committee to join the committee...")
        node:SentraNode
        grpcNode:SentraBackend_GRPC_Services_pb2.t_grpc_SentraNode=SentraBackend_GRPC_Services_pb2.t_grpc_SentraNode()
        req:SentraBackend_GRPC_Services_pb2.JoinCommitteeRequest=SentraBackend_GRPC_Services_pb2.JoinCommitteeRequest()
        for node in committee.getNodes():
            grpcNode.node_id=node.m_strNodeID
            grpcNode.grpc_url=node.m_strInterNodeCommunicationGRPC_URL
            req.committee.add().CopyFrom(grpcNode)
        grpcMsg=SentraBackend_GRPC_Services_pb2.ServerMessage(join_committee_request=req)
        for node in committee.getNodes():
            log(f"Send committee join to node: {node.m_strNodeID}")
            self.putMessageInNodeSendQueue(node.m_sendQueue,grpcMsg)


    async def generateComittee(self):
        if(self.m_nodeList.len()>=self.m_commandLineOptions.getCommitteeSelectionTrigger()) and not self.m_bCommitteeSelected:
            committee_target_size = 5
            committee_min_trust = 2
            committee_max_attest_age = 100
            committee_max_hw_frac = 0.8
            committee_max_op_frac = 0.8
            committee_max_pm_frac = 0.8

            committeeSelection: CommitteeSelection = CommitteeSelection(committee_target_size, committee_min_trust, committee_max_attest_age, committee_max_hw_frac, committee_max_op_frac, committee_max_pm_frac)

            log(f"running committee selction with: committee target size: {committee_target_size}, committee min trust: {committee_min_trust}, committee_max_attest_age: {committee_max_attest_age}, committee_max_hw_frac: {committee_max_hw_frac}, committee_max_op_frac: {committee_max_op_frac}, committee_max_pm_frac: {committee_max_pm_frac}")
            committee = committeeSelection.selectionAlgorithm(self.m_nodeList)

            if committee:
                log("found committee:")
                log(str(committee))
                await self.closeTcpProxies()
                self.m_sortedCommittee = sorted(committee.m_arNodes.values(), key=lambda node: node.m_fTrustScore)
                log("open TCP proxies")
                await self.openTcpProxies()
                await self.requestCommitteeJoin(committee)
                self.m_bCommitteeSelected=True
            else:
                log("no committee found!")

    # ------------------------------------------------------------------
    # High-level: open/close all ports for an entire committee at once
    # ------------------------------------------------------------------

    async def openTcpProxies(self) -> None:
        """
        Open one TCP port for every node in the committee.
        Called after a new committee has been selected.
        """
        if(self.m_sortedCommittee is None):
            return
        for index, node in enumerate(self.m_sortedCommittee):
            await self.openPortForNode(index, node.m_sendQueue)

    async def closeTcpProxies(self) -> None:
        """
        Close the TCP port for every node in the committee.
        Called before a new committee replaces the current one.
        """
        if(self.m_sortedCommittee is None):
            return
        for index, node in enumerate(self.m_sortedCommittee):
            self._send_queues[index]
            await self.closePortForNode(index)

    # ------------------------------------------------------------------
    # Called from NodeMessageServiceServicer when a node registers
    # ------------------------------------------------------------------

    async def openPortForNode(
        self,
        committee_index: int,
        send_queue: asyncio.Queue,
    ) -> None:
        """
        Open a TCP listener on base_port + committee_index for the given node.
        Already called from within the gRPC event loop, so plain await is fine.
        """
        if committee_index in self._servers:
            log(f"[TCPProxy] Port already open for committee index {committee_index}, skipping")
            return

        port = self.base_port + committee_index
        self._send_queues[committee_index] = send_queue

        server = await asyncio.start_server(
            lambda r, w: self._handle_tcp_client(committee_index, r, w),
            host="0.0.0.0",
            port=port,
        )
        self._servers[committee_index] = server
        asyncio.create_task(server.serve_forever())

        log(f"[TCPProxy] Opened TCP port {port} for committee member {committee_index}")

    async def closePortForNode(self, committee_index: int) -> None:
        """
        Close the TCP listener and any active TCP client for this committee member.
        """
        await self.m_tcpProxy.closePortForNode(committee_index)

    async def NodeStream(self, request_iterator, context) -> None:
        node_id: str | None = None
        bRegistered: bool = False
        log("New streaming connection established")

        # First message has to be a register message
        try:
            first_message:object|None = await anext(request_iterator, None)
            if first_message is None:
                log("Connection closed before registration")
                return

            resp, node_id,send_queue = self.registerNode(first_message)
            if node_id is not None:
                yield SentraBackend_GRPC_Services_pb2.ServerMessage(response=resp)
                req = self.generateAttestationRequest()
                yield SentraBackend_GRPC_Services_pb2.ServerMessage(attestation=req)
                bRegistered = True
        except Exception as e:
            log(f"Error during registration: {e}")
            return

        if not bRegistered or node_id is None or send_queue is None:
            log("New Node not registered - closing connection")
            return

        async def recv_messages():
            '''Internal function to receive GRPC messages from Sentra Nodes'''
            try:
                # Process remaining messages
                async for node_message in request_iterator:
                    if node_message.HasField('quote'):
                        log("Received quote")
                        attestation = Attestation()
                        bVerified: bool = False
                        if(self.m_commandLineOptions.getAcceptFakeAttestation()):
                            bVerified=True
                        else:
                            bVerified=await attestation.verify(node_message.quote.report)
                        if bVerified:
                            attest_time = time.time()
                            self.m_nodeList.setVerified(node_id, attest_time)
                            log(f"Node {node_id} verified.")
                            await self.generateComittee()
                        else:
                            log(f"Node {node_id} failed verification")
                            break
                    elif node_message.HasField('python_msg'):
                        log(f"Received python_msg from node {node_id}, TODO forward to TCP proxy")

                        node: SentraNode | None = self.m_nodeList.getNode(node_id)
                        committee_index = self.m_sortedCommittee.index(node)
                        await self.m_tcpProxy.forwardToTCPClient(
                            committee_index, node_message.python_msg.msg
                        )
                    else:
                        log(f"Unexpected message type from node {node_id}")
                        break
            except Exception as e:
                log(f"Error processing messages from node {node_id}: {e}")
            finally:
                self.m_nodeList.remove(node_id)
                await self.m_tcpProxy.closePortForNode(node_id)
                log(f"Leaving receive loop closing connection to node {node_id}...")

        recv_task:asyncio.Task[object]=asyncio.create_task(recv_messages())
        try:
            while True:
                message = await send_queue.get()
                if message is None:  # Shutdown signal
                    break
                yield message
        except Exception as e:
            log(f"Error sending to {node_id}: {e}")
        finally:
            recv_task.cancel()
            log(f"Leaving send loop closing connection to node {node_id}...")
