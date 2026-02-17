import argparse
import json
import socket
from flask import Flask, jsonify
from flask.json.provider import DefaultJSONProvider
from flask import send_file

import grpc.aio as grpc_aio
import threading
import SentraBackend_GRPC_Services_pb2
import SentraBackend_GRPC_Services_pb2_grpc

import time
import sys
import dcap_qvl
import asyncio
import random
from collections import defaultdict
import copy

#For getting TLS certificate using ACME
import josepy as jose
import acme.client
import acme.messages
from acme import challenges
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives import serialization
from acme import crypto_util

BACKEND_VERSION="00.04.017"


# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def log(message:str):
    """Log with immediate flush for Docker visibility."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

class CommandLineOptions:

    m_Parser=argparse.ArgumentParser("Sentra Backend",
                                     "This programm is the Sentra backend. It manages the Sentra nodes and the related training/inference process.",
                                     "Below are the valid command line options.")

    def __init__(self):
        self.m_Parser.add_argument("-p","--port",help="Port to listen on for Web/REST-API requests",default=8888)
        self.m_Parser.add_argument("-l","--host",help="Host to listen on for Web/REST-API requests",default="127.0.0.1")
        self.m_Parser.add_argument("-s","--simulator",help="Simulate the Nodes to allow for some UI testing.",default=False,action="store_true")
        self.m_Parser.add_argument("-c","--committee",help="Run the committee selection using the simulation.",default=False,action="store_true")
        self.m_Parser.add_argument("--committee-size",help="Size of the committee.",default=0,type=int)
        self.m_Parser.add_argument("--committee-selection-trigger",help="Number of available nodes which trigger the commitee selection.",default=0,type=int)
        self.m_Parser.add_argument("--use-acme",help="Use ACME to get a certificate for the GRPC interface (otherwise plain HTTP is used).",default=False,action="store_true")
        self.m_Parser.add_argument("--acme-host",help="Host of the ACME server.",default="localhost")
        self.m_Parser.add_argument("--acme-server-certificate",help="Path to the CA certificate for verifying TLS connections with the ACME server. If not given, the TLS connection will not be verified.",default=None)
        self.m_Args=self.m_Parser.parse_args()

    def getPort(self)->int:
        return self.m_Args.port

    def getHost(self)->str:
        return self.m_Args.host

    def getRunInSimulationMode(self)->bool:
        return self.m_Args.simulator

    def getRunCommitteeSelection(self)->bool:
        return self.m_Args.committee

    def useACME(self)->bool:
        return self.m_Args.use_acme

    def getACMEHost(self)->str:
        return self.m_Args.acme_host

    def getACMEServerCertificate(self)->str|None:
        return self.m_Args.acme_server_certificate

    def getComitteeSelectionTrigger(self)->int:
        return self.m_Args.comittee_selection_trigger

class SentraACME:

    m_acmeHost:str
    m_acmeCert:str|None
    def __init__(self,acmeHost:str,acmeCert:str|None):
        self.m_acmeCert=acmeCert
        self.m_acmeHost=acmeHost

    def generateTLSCertsAndKeys(self):
        try:
            log("Try to get TLS certificate...")
            acc_key = jose.JWKRSA(key=rsa.generate_private_key(65537, 2048, default_backend()))
            # Connect and register (single account creation)
            acmeCert:str|bool
            if(self.m_acmeCert is None):
                acmeCert=False
            else:
                acmeCert=self.m_acmeCert
            net = acme.client.ClientNetwork(acc_key, verify_ssl=acmeCert)
            strURL:str=f"https://{self.m_acmeHost}:14000/dir"
            directory = acme.client.ClientV2.get_directory(strURL, net)
            _acme:acme.client.ClientV2 = acme.client.ClientV2(directory, net=net)
            _acme.new_account(acme.messages.NewRegistration.from_data(terms_of_service_agreed=True))

            # Generate private key for certificate
            cert_key:RSAPrivateKey = rsa.generate_private_key(65537, 2048, default_backend())
            key_pem:bytes = cert_key.private_bytes(serialization.Encoding.PEM,
                                            serialization.PrivateFormat.TraditionalOpenSSL,
                                                serialization.NoEncryption()
                                            )
            hostname:str=socket.gethostname()
            csr_pem:bytes = crypto_util.make_csr(key_pem, [hostname])
            # Order certificate
            order:acme.messages.OrderResource = _acme.new_order(csr_pem)
            for authz in order.authorizations:
                # Try to find a supported challenge type
                challenge = None

                for chall in authz.body.challenges:
                    if isinstance(chall.chall, challenges.HTTP01):
                        challenge = chall
                        break
                    elif isinstance(chall.chall, challenges.DNS01):
                        challenge = chall
                        break

                if challenge:
                    response = challenge.response(acc_key)
                    _acme.answer_challenge(challenge, response)

            order = _acme.poll_and_finalize(order)
            log("Got TLS certificate.")
            return order
        except:
            log("Failure in getting TLS certificate - continue without TLS.")
            return None

class Attestation:
    async def verify(self,quote:bytes)->bool:
        # Verify the quote
        log("try to verify quote...")
        now_timestamp = int(time.time())
        result:dcap_qvl.VerifiedReport
        try:
            collateral =  await dcap_qvl.get_collateral_from_pcs(quote)
            result =  dcap_qvl.verify(quote,collateral,now_timestamp)
        except Exception as e:
            log("Sone excepetion in verify")
            log(str(e))
            return False
        log(result.status)
        #ToDo - need to check the result

        parsed_quote:dcap_qvl.Quote = dcap_qvl.parse_quote(quote)
        quote_header:dcap_qvl.QuoteHeader=parsed_quote.header
        log(str(quote_header.version))
        log(str(quote_header.attestation_key_type))
        log(quote_header.user_data.hex())

        enclave_report = parsed_quote.report
        if(not isinstance(enclave_report,dcap_qvl.SgxEnclaveReport)):
            log("Not an SGX enclave")
            return False
        sgx_report:dcap_qvl.SgxEnclaveReport =enclave_report
        # Zugriff auf die wichtigsten Felder
        log(f"MRENCLAVE: {sgx_report.mr_enclave.hex()}")
        log(f"MRSIGNER:  {sgx_report.mr_signer.hex()}")
        log(f"Attributes: {sgx_report.attributes.hex()}")
        log(f"ReportData: {sgx_report.report_data.hex()}")
        return True

class SentraNode:
    m_strNodeID:str
    m_iCPUArchitecture:int
    m_iOperator:int
    m_iHost:int
    m_fTrustScore:float
    m_attestTime: float
    m_bVerified:bool
    m_sendQueue: asyncio.Queue[object]

    def __init__(self,nodeID:str,trustscore:float,cpu:int,host:int,operator:int):
        self.m_iOperator=operator
        self.m_fTrustScore=trustscore
        self.m_iCPUArchitecture=cpu
        self.m_iHost=host
        self.m_strNodeID=nodeID
        self.m_bVerified=False
        self.m_attestTime=0
        self.m_sendQueue= asyncio.Queue()

    def __hash__(self):
        return hash(self.m_strNodeID)

    def __eq__(self, other:object)->bool:
        if not isinstance(other, SentraNode):
            return NotImplemented
        return self.m_strNodeID == other.m_strNodeID

    def to_dict(self):
        return {"nodeID":self.m_strNodeID,
                "cpu arch":self.m_iCPUArchitecture,
                "operator":self.m_iOperator,
                "host":self.m_iHost,
                "trust score":self.m_fTrustScore,
                "attest time":self.m_attestTime,
                "verified":self.m_bVerified}

    def setVerified(self,b:bool,time:float)->None:
        self.m_bVerified=b
        self.m_attestTime=time

    def getSendQueue(self)-> asyncio.Queue[object]:
        return self.m_sendQueue

    def toJSONObject(self)->object:
            return {
                'node_id':self.m_strNodeID,
                'host':backend.m_nodeGenerator.getHost(self.m_iHost).m_Name,
                'operator':backend.m_nodeGenerator.getOpertor(self.m_iOperator),
                'cpu':backend.m_nodeGenerator.getCPUForHost(self.m_iHost),
                'attested':self.m_bVerified
            }

class Host:
    m_Name:str
    m_iCPU:int

ARCHITECTURES = 3
HOSTS = 10
OPERATORS = 10

class SentraNodeAttributeGenerator:
    m_minTrustScore:float
    m_maxTrustScore:float
    m_arCPUArchitectures:list[str]
    m_numCPUArchitectures:int
    m_numHosts:int
    m_arHosts:list[Host]
    m_numOperators:int
    m_arOperators:list[str]

    def __init__(self,minTrustScore:float,maxTrustScore:float,
                 cpuArchitectures:list[str]=["Intel","AMD","ARM"],
                 numOperators:int=OPERATORS,numHosts:int=HOSTS):
        self.m_arCPUArchitectures=cpuArchitectures
        self.m_numCPUArchitectures=len(self.m_arCPUArchitectures)
        self.m_minTrustScore=minTrustScore
        self.m_maxTrustScore=maxTrustScore
        self.m_numHosts=numHosts
        self.internal_generateAttestTime()
        self.internal_generateHosts()
        self.m_numOperators=numOperators
        self.internal_generateOperators()

    def internal_generateHosts(self)->None:
        i:int=0
        self.m_arHosts:list[Host]=[]
        while(i<self.m_numHosts):
            h:Host=Host()
            h.m_Name="Host "+str(i)
            h.m_iCPU=self.internal_generateCPU()
            self.m_arHosts.append(h)
            i+=1

    def internal_generateOperators(self)->None:
        i:int=0
        self.m_arOperators:list[str]=[]
        while(i<self.m_numOperators):
            self.m_arOperators.append("Operator "+str(i))
            i+=1

    def internal_generateCPU(self)->int:
        return random.randrange(self.m_numCPUArchitectures)

    def internal_generateOperator(self)->int:
        return random.randrange(self.m_numOperators)

    def internal_generateTrustScore(self)->float:
        return random.uniform(self.m_minTrustScore, self.m_maxTrustScore)

    def internal_generateHost(self)->int:
        return random.randrange(self.m_numHosts)

    def internal_generateAttestTime(self):
        self.m_attestTime = time.time()

    def getHost(self,i:int)->Host:
        return self.m_arHosts[i]

    def getOpertor(self,i:int)->str:
        return self.m_arOperators[i]

    def getCPU(self,i:int)->str:
        return self.m_arCPUArchitectures[i]

    def getCPUForHost(self,i:int)->str:
        return self.getCPU(self.m_arHosts[i].m_iCPU)

    def generateNode(self,nodeID:str) -> SentraNode:
        host:int=self.internal_generateHost()
        operator:int=self.internal_generateOperator()
        trust=self.internal_generateTrustScore()
        cpu=self.getHost(host).m_iCPU
        attestTime = self.m_attestTime
        node:SentraNode=SentraNode(nodeID,trust,cpu,host,operator)
        return node

class SentraNodeList:
    m_arNodes:dict[str,SentraNode]={}
    m_Lock:threading.Lock

    def __init__(self):
        self.m_arNodes={}
        self.m_Lock=threading.Lock()

    def add(self,node:SentraNode)->bool:
        with self.m_Lock:
            if(node.m_strNodeID in self.m_arNodes):
                return False
            self.m_arNodes[node.m_strNodeID]=node
            return True

    def remove(self,node_id:str|None):
        if(not node_id is None):
            with self.m_Lock:
                self.m_arNodes.pop(node_id,None)

    def setVerified(self,node_id:str,time:float)->bool:
        with self.m_Lock:
            node:SentraNode|None=self.m_arNodes.get(node_id,None)
            if(not node is None):
                node.setVerified(True,time)
                return True
        return False

    def getSendQueue(self,node_id:str)-> asyncio.Queue[object]|None:
        with self.m_Lock:
            node:SentraNode|None=self.m_arNodes.get(node_id,None)
            if(not node is None):
                return node.getSendQueue()
        return None

    def len(self)->int:
        with self.m_Lock:
            return len(self.m_arNodes)

    def toJSONObject(self)->object:
        with self.m_Lock:
            return [*self.m_arNodes.values()]

class Comittee:
    m_Comittee:set[str]
    def __init__(self):
     self.m_Comittee=set()

    def addComitteeMember(self,node_id:str):
        self.m_Comittee.add(node_id)


class CommitteeSelection:
    """
    Implements the committee selection algorithm of Sentra.

    Attributes:
        m_old_committee (SentraNodeList): The committee of the previous epoch,
        if exists.
        m_target_size (int): Target size of the committee.
        m_min_trust (float): Required minimal trust score of a node to be
        considered as candidate.
        m_max_attest_age (int): Maximum time after last attestation.
        m_max_hw_frac (float): Diversity constraint 1 - maximum fraction of
        nodes in the committee running on the same hardware platform.
        m_max_op_frac (float): Diversity constraint 2 - maximum fraction of
        nodes in the committee run by the same operator.
        m_max_pm_frac (float): Diversity constraint 3 - maximum fraction of
        nodes in the committee running on the same CPU.
    """

    m_old_committee: SentraNodeList = SentraNodeList()
    m_target_size: int
    m_min_trust: float
    m_max_attest_age: int
    m_max_hw_frac: float
    m_max_op_frac: float
    m_max_pm_frac: float

    def __init__(self, target_size: int, min_trust: int, max_attest_age: int,
                 max_hw_frac: float, max_op_frac: float, max_pm_frac: float):
        """
        Initializes the CommitteeSelection.

        Args:
            target_size (int): Value stored in the node.
            min_trust (int): Minimal trust score of a node to be considered as
            candidate.
            max_attest_age (int): Maximum time after last attestation.
            max_hw_frac (int): Maximum fraction of nodes in the committee
            running on the same hardware platform.
            max_op_frac (int): Maximum fraction of nodes in the committee run
            by the same operator.
            max_pm_frac (int): Maximum fraction of nodes in the committee
            running on the same CPU.
        """

        if not all(0 <= x <= 1 for x in (max_hw_frac, max_op_frac, max_pm_frac)):
            raise ValueError(
                "max_hw_frac, max_op_frac, max_pm_frac must be in range 0-1")

        if not all(x > 0 for x in (target_size, min_trust, max_attest_age)):
            raise ValueError(
                "target_size, min_trust, max_attest_age must be \
                greater then zero")

        self.m_target_size = target_size
        self.m_min_trust = min_trust
        self.m_max_attest_age = max_attest_age
        self.m_max_hw_frac = max_hw_frac
        self.m_max_op_frac = max_op_frac
        self.m_max_pm_frac = max_pm_frac

    def filter(self, nodes: SentraNodeList) -> SentraNodeList:
        """
        Step 1 of node selection algorithm.

        Filter the candidate list for all nodes which fulfill the attestation
        requirements.

        Args:
            nodes (SentraNodeList): Available nodes.

        Returns:
            SentraNodeList: Candidate nodes.
        """

        curr_time = time.time()

        # have to create a copy for deleting while iterating
        for node in list(nodes.m_arNodes.values()):
            if not node.m_bVerified:
                log(f"removing node {node.m_strNodeID} from candidate set, \
                     reason: not verified")
                nodes.remove(node.m_strNodeID)
                continue

            if node.m_fTrustScore < self.m_min_trust:
                log(f"removing node {node.m_strNodeID} from candidate set, \
                     reason: trustScore too low")
                nodes.remove(node.m_strNodeID)
                continue

            if curr_time - node.m_attestTime > self.m_max_attest_age:
                log(f"removing node {node.m_strNodeID} from candidate set, \
                    reason: attestation too old")
                nodes.remove(node.m_strNodeID)
                continue

        return nodes

    def check_reuse(self):
        """
        Step 2 of node selection algorithm.

        Check, if the previous committee (if existent) can be reused.

        Returns:
            Bool: If old committee can be reused.
        """

        if all(node_id in self.m_candidates.m_arNodes
               for node_id in self.m_old_committee.m_arNodes):

            log("previous committee can be reused")
            return True

        else:

            log("previous committee can't be reused")
            return False

    # TODO: so far only returns the existing value.
    def score(self, node: SentraNode):
        """
        Step 3 of node selection algorithm.

        Compute the score of a single node.

        Args:
            node (SentraNode): Node to compute the trust score of.

        Returns:
            float: Trust score of the node.
        """

        return node.m_fTrustScore

    def greedy_select(self, sorted_candidates):
        """
        Step 4 of node selection algorithm.

        Select the committee for epoch e+1 from the sorted candidate list using a greedy algorithm. Selection is based on the configured diversity constraints.

        Args:
            sorted_candidates (SentraNodeList): Sorted list of candidate nodes.

        Returns:
            SentraNodeList: Node list of new committee. If length == 0, no committee was found.
        """

        count_hw = defaultdict(int)
        count_op = defaultdict(int)
        count_pm = defaultdict(int)

        new_committee = SentraNodeList()

        for node in sorted_candidates:
            node_hw = node.m_iCPUArchitecture
            node_op = node.m_iOperator
            node_pm = node.m_iHost
            frac_hw_new = (count_hw.get(node_hw, 0) + 1) / self.m_target_size
            frac_op_new = (count_op.get(node_op, 0) + 1) / self.m_target_size
            frac_pm_new = (count_pm.get(node_pm, 0) + 1) / self.m_target_size

            if (
                frac_hw_new <= self.m_max_hw_frac
                and frac_op_new <= self.m_max_op_frac
                and frac_pm_new <= self.m_max_pm_frac
            ):

                count_hw[node_hw] = count_hw.get(node_hw, 0) + 1
                count_op[node_op] = count_op.get(node_op, 0) + 1
                count_pm[node_pm] = count_pm.get(node_pm, 0) + 1
                new_committee.add(node)

                log(f"node {node.m_strNodeID} fulfills requirements")

                if new_committee.len() == self.m_target_size:
                    return new_committee

        return new_committee

    def selectionAlgorithm(self, nodes: SentraNodeList) -> SentraNodeList:
        """
        Node selection algorithm.

        Select the committee for epoch e+1 from a list of available nodes.

        Args:
            nodes (SentraNodeList): List of available nodes.

        Returns:
            SentraNodeList: Node list of new committee. If length == 0,
            no committee was found.
        """

        log("select nodes for committee")
        self.m_candidates = self.filter(nodes)

        new_committee: SentraNodeList = SentraNodeList()

        if self.m_candidates.len() < self.m_target_size:
            log(f"candidate list size after filtering: \
                {self.m_candidates.len()} is smaller than required \
                committee size: {self.m_target_size}")
            return

        if self.m_old_committee.len():
            if (self.check_reuse()):
                return self.m_old_committee

        sorted_candidates = sorted(self.m_candidates.m_arNodes.values(),
                                   key=self.score)

        new_committee = self.greedy_select(sorted_candidates)

        if not new_committee.len():
            log("no committee found")
            return new_committee

        else:
            log(f"found committee of size {new_committee.len()}")

        self.m_old_committee = new_committee

        return new_committee


class NodeMessageServiceServicer(SentraBackend_GRPC_Services_pb2_grpc.NodeMessageServiceServicer):

    m_nodeGenerator:SentraNodeAttributeGenerator
    m_nodeList:SentraNodeList
    m_GRPC_Loop:asyncio.AbstractEventLoop

    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeList:SentraNodeList):
        self.m_nodeGenerator=nodeGenerator
        self.m_nodeList=nodeList
        self.m_GRPC_Loop=asyncio.get_event_loop()

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

    def generateComittee(self):
        if(self.m_nodeList.len()>=10):
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
                self.generateComittee()
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
                        bVerified: bool = await attestation.verify(node_message.quote.report)
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

class AppSimulator:

    m_Thread:threading.Thread|None
    m_nodeGenerator:SentraNodeAttributeGenerator

    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeList:SentraNodeList, committeeSelection):
        self.m_nodeGenerator=nodeGenerator
        self.m_nodeList=nodeList
        self.m_committeeSelection = committeeSelection

    def runSimulation(self):
        i:int=0
        baseId:str="SentraNode_"
        while(i<10):
            node:SentraNode=self.m_nodeGenerator.generateNode(baseId+str(i))
            if(random.random()>0.1):
                node.setVerified(True, time.time())
            self.m_nodeList.add(node)
            i+=1

        if self.m_committeeSelection:
            log("node list:")
            json_list = [node.to_dict()
                         for node in self.m_nodeList.m_arNodes.values()]
            log(json.dumps(json_list, indent=4))

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
            else:
                log("no committee found!")

    def start(self):
        self.m_Thread = threading.Thread(target=self.runSimulation, args=(), daemon=True)
        self.m_Thread.start()

class CustomJSONProvider(DefaultJSONProvider):

    def default(self, obj:object)->object:
        if isinstance(obj, SentraNode) or isinstance(obj,SentraNodeList):
            return obj.toJSONObject()
        return super().default(obj)

class Backend:

    m_sStaticFolder="../frontend/dist/frontend/browser"
    m_sIndexHtml=m_sStaticFolder+"/index.html"
    m_nodeGenerator:SentraNodeAttributeGenerator
    m_nodeList:SentraNodeList
    m_bAppSimulation:bool=False
    m_appSimulator:AppSimulator|None=None
    m_NodeMessageServiceServicer:NodeMessageServiceServicer

    def __init__(self):
        self.m_bAppSimulation=False
        self.m_appSimulator=None

    async def runGRPCServer(self):
        self.server = grpc_aio.server()
        self.m_NodeMessageServiceServicer = NodeMessageServiceServicer(self.m_nodeGenerator,self.m_nodeList)
        SentraBackend_GRPC_Services_pb2_grpc.add_NodeMessageServiceServicer_to_server(self.m_NodeMessageServiceServicer,
                                                                                       self.server)
        self.server.add_insecure_port('0.0.0.0:8000')
        log(f"Starting gRPC server on port 8000...")
        await self.server.start()
        await self.server.wait_for_termination()

    def startGRPCServer(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.runGRPCServer())
        finally:
            loop.close()

    def createGRPCServer(self):
#        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        grpc_thread = threading.Thread(target=self.startGRPCServer, args=(), daemon=True)
        grpc_thread.start()

    def create(self,cmdlineargs:CommandLineOptions)->Flask:
        self.m_bAppSimulation=cmdlineargs.getRunInSimulationMode()
        self.m_committee_selection=cmdlineargs.getRunCommitteeSelection()
        self.app:Flask = Flask(__name__,static_url_path='',static_folder=self.m_sStaticFolder)
        self.app.add_url_rule("/",view_func=self.getIndex)
        self.app.add_url_rule("/api/v1/getNodes",view_func=self.getNodes)
        self.app.json = CustomJSONProvider(self.app)

        self.m_nodeGenerator=SentraNodeAttributeGenerator(1.0,10.0)
        self.m_nodeList=SentraNodeList()
        if(self.m_bAppSimulation):
            log("Enable Sentra Node Simulation")
            self.m_appSimulator=AppSimulator(self.m_nodeGenerator,self.m_nodeList, self.m_committee_selection)
            self.m_appSimulator.start()

        self.createGRPCServer()
        return self.app

    def sendMessageToNode(self,node_id:str,message:object)->None:
        self.m_NodeMessageServiceServicer.sendMessageToNode(node_id, message)


    def getIndex(self):
        return send_file(self.m_sIndexHtml)

    #@app.route('/api/v1/getNodes', methods=['GET'])
    def getNodes(self):
        return jsonify(self.m_nodeList.toJSONObject())

backend:Backend

if __name__ == '__main__':
    log("Starting Sentra Backend...")
    log(f"Version: {BACKEND_VERSION}")

    cmdlineargs=CommandLineOptions()
    if(cmdlineargs.useACME()):
        acme_connection:SentraACME=SentraACME(cmdlineargs.getACMEHost(),cmdlineargs.getACMEServerCertificate())
        acme_connection.generateTLSCertsAndKeys()

    backend=Backend()
    app:Flask=backend.create(cmdlineargs)
    app.run(debug=False,port=cmdlineargs.getPort(),host=cmdlineargs.getHost(),threaded=True)
