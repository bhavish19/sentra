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

class NodeMessageServiceServicer(_NodeMessageServiceServicer):

    m_nodeGenerator:SentraNodeAttributeGenerator
    m_nodeList:SentraNodeList
    m_GRPC_Loop:asyncio.AbstractEventLoop
    m_commandLineOptions:CommandLineOptions

    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeList:SentraNodeList,commandlineOptions:CommandLineOptions):
        self.m_nodeGenerator=nodeGenerator
        self.m_nodeList=nodeList
        self.m_GRPC_Loop=asyncio.get_event_loop()
        self.m_commandLineOptions=commandlineOptions

    def sendMessageToNode(self,node_id:str,message:object)->None:
        sendQueue: asyncio.Queue[object]|None=self.m_nodeList.getSendQueue(node_id)
        if(not sendQueue is None):
            asyncio.run_coroutine_threadsafe(sendQueue.put(message),self.m_GRPC_Loop)

    def registerNode(self, message:object)->tuple[object,str|None,asyncio.Queue[object]|None]:
        if not message.HasField('register'):
            return (None,None,None)
        register_message:object=message.register
        node_id:str|None = register_message.node_id

        if not node_id or not isinstance(node_id,str):
            return (SentraBackend_GRPC_Services_pb2.RegisterResponse(
                success=False,
                message="Node ID cannot be empty"
            ),None,None)
        node:SentraNode=self.m_nodeGenerator.generateNode(node_id)
        if(self.m_nodeList.add(node)):
            log(f"Node registered: {node_id} - Node list now has {self.m_nodeList.len()} entries")
            return (SentraBackend_GRPC_Services_pb2.RegisterResponse(
                success=True,
                message=f"Node {node_id} registered successfully"
            ),node_id,node.getSendQueue())
        else:
            return (SentraBackend_GRPC_Services_pb2.RegisterResponse(
                success=True,
                message=f"Node {node_id} already registered"
            ),None,None)


    def generateAttestationRequest(self)->SentraBackend_GRPC_Services_pb2.AttestationRequest:
        return SentraBackend_GRPC_Services_pb2.AttestationRequest(nonce="Nonce")

    async def requestCommitteeJoin(self,committee:SentraNodeList):
        for node in committee.getNodes():
            req:SentraBackend_GRPC_Services_pb2.JoinCommitteeRequest=SentraBackend_GRPC_Services_pb2.JoinCommitteeRequest(node_id=node.node_id,host=node.host,port=node.port)
            await node.getSendQueue().put(req)


    async def generateComittee(self):
        if(self.m_nodeList.len()>=self.m_commandLineOptions.getCommitteeSelectionTrigger()):
            committee_target_size = 5
            committee_min_trust = 2
            committee_max_attest_age = 100
            committee_max_hw_frac = 0.8
            committee_max_op_frac = 0.8
            committee_max_pm_frac = 0.8

            committeeSelection: CommitteeSelection = CommitteeSelection(committee_target_size, committee_min_trust, committee_max_attest_age, committee_max_hw_frac, committee_max_op_frac, committee_max_pm_frac)

            committee = committeeSelection.selectionAlgorithm(self.m_nodeList)

            log(f"running committee selction with: committee target size: {committee_target_size}, committee min trust: {committee_min_trust}, committee_max_attest_age: {committee_max_attest_age}, committee_max_hw_frac: {committee_max_hw_frac}, committee_max_op_frac: {committee_max_op_frac}, committee_max_pm_frac: {committee_max_pm_frac}")

            if committee:
                log("committee:")
                json_list = [node.to_dict() for node in committee.m_arNodes.values()]
                log(json.dumps(json_list, indent=4))
                await self.requestCommitteeJoin(committee)
            else:
                log("no committee found!")

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
                await self.generateComittee()
        except Exception as e:
            log(f"Error during registration: {e}")
            return

        if not bRegistered or node_id is None or send_queue is None:
            log("New Node not registered - closing connection")
            return

        async def recv_messages():
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
                        else:
                            log(f"Node {node_id} failed verification")
                            break
                    else:
                        log(f"Unexpected message type from node {node_id}")
                        break
            except Exception as e:
                log(f"Error processing messages from node {node_id}: {e}")
            finally:
                self.m_nodeList.remove(node_id)
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
