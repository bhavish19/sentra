import argparse
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

BACKEND_VERSION="00.03.011"


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
        self.m_Args=self.m_Parser.parse_args()
        
    def getPort(self)->int:
        return self.m_Args.port

    def getHost(self)->str:
        return self.m_Args.host
    
    def getRunInSimulationMode(self)->bool:
        return self.m_Args.simulator


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
    m_bVerified:bool
    m_sendQueue: asyncio.Queue[object]

    def __init__(self,nodeID:str,trustscore:float,cpu:int,host:int,operator:int):
        self.m_iOperator=operator
        self.m_fTrustScore=trustscore
        self.m_iCPUArchitecture=cpu
        self.m_iHost=host
        self.m_strNodeID=nodeID
        self.m_bVerified=False
        self.m_sendQueue= asyncio.Queue()
    
    def __hash__(self):
        return hash(self.m_strNodeID)
    
    def __eq__(self, other:object)->bool:
        if not isinstance(other, SentraNode):
            return NotImplemented 
        return self.m_strNodeID == other.m_strNodeID
    
    def setVerified(self,b:bool)->None:
        self.m_bVerified=b

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
                 numOperators:int=10,numHosts:int=10):
        self.m_arCPUArchitectures=cpuArchitectures
        self.m_numCPUArchitectures=len(self.m_arCPUArchitectures)
        self.m_minTrustScore=minTrustScore
        self.m_maxTrustScore=maxTrustScore
        self.m_numHosts=numHosts
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

    def setVerified(self,node_id:str)->bool:
        with self.m_Lock:
            node:SentraNode|None=self.m_arNodes.get(node_id,None)
            if(not node is None):
                node.setVerified(True)
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
            try:
                # Process remaining messages
                async for node_message in request_iterator:
                    if node_message.HasField('quote'):
                        log("Received quote")
                        attestation = Attestation()
                        bVerified: bool = await attestation.verify(node_message.quote.report)
                        if bVerified:
                            self.m_nodeList.setVerified(node_id)
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
    m_nodeList:SentraNodeList

    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeList:SentraNodeList):
        self.m_nodeGenerator=nodeGenerator
        self.m_nodeList=nodeList

    def runSimulation(self):
        i:int=0
        baseId:str="SentraNode_"
        while(i<10):
            node:SentraNode=self.m_nodeGenerator.generateNode(baseId+str(i))
            if(random.random()>0.1):
                node.setVerified(True)
            self.m_nodeList.add(node)
            i+=1        

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
        self.app:Flask = Flask(__name__,static_url_path='',static_folder=self.m_sStaticFolder)
        self.app.add_url_rule("/",view_func=self.getIndex)
        self.app.add_url_rule("/api/v1/getNodes",view_func=self.getNodes)
        self.app.json = CustomJSONProvider(self.app)

        self.m_nodeGenerator=SentraNodeAttributeGenerator(1.0,10.0)
        self.m_nodeList=SentraNodeList()
        if(self.m_bAppSimulation):
            log("Enable Sentra Node Simulation")
            self.m_appSimulator=AppSimulator(self.m_nodeGenerator,self.m_nodeList)
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
    backend=Backend()
    app:Flask=backend.create(cmdlineargs)
    app.run(debug=False,port=cmdlineargs.getPort(),host=cmdlineargs.getHost(),threaded=True)
