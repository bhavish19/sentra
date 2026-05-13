from flask import Flask, jsonify, request
from flask import send_file
import threading
import asyncio
import grpc
import grpc.aio as grpc_aio
from simple_websocket import Server

from .CommandLineOptions import CommandLineOptions
from .CustomJSONProvider import CustomJSONProvider
from .SentraNodeAttributeGenerator import SentraNodeAttributeGenerator
from .SentraNodeList import SentraNodeList
from .AppSimulator import AppSimulator
from .grpc import NodeMessageServiceServicer
from .SentraACME import SentraACME
from .Log import log as log
from .grpc import add_NodeMessageServiceServicer_to_server
from .grpc import add_TCPProxyServiceServicer_to_server
from .ClientDistributor import ClientDistributor
from .SentraML import InferenceResult
from .MNIST import MNIST

class Backend:
    theBackend=None

    m_sStaticFolder="../../frontend/dist/frontend/browser"
    m_sIndexHtml=m_sStaticFolder+"/index.html"
    
    m_WS:Server|None = None
    m_WSClient:Server|None = None
    m_lockWS=threading.Lock()
    m_lockWSClient=threading.Lock()


    m_nodeGenerator:SentraNodeAttributeGenerator
    m_nodeList:SentraNodeList
    m_Committee:SentraNodeList|None
    m_bAppSimulation:bool=False
    m_appSimulator:AppSimulator|None=None
    m_NodeMessageServiceServicer:NodeMessageServiceServicer
    m_grpcCertsPEM:bytes|None
    m_grpcKeyPEM:bytes|None
    m_commandLineOptions:CommandLineOptions
    m_clientDistributor: ClientDistributor
    m_MNIST:MNIST

    def __init__(self):
        self.__class__.theBackend=self
        self.m_bAppSimulation=False
        self.m_appSimulator=None
        self.m_grpcCertsPEM=None
        self.m_grpcKeyPEM=None
        self.m_Committee=None
        self.m_MNIST=MNIST()

    @classmethod
    def getBackend(cls) -> "Backend|None":
        return cls.theBackend

    async def runGRPCServer(self):
        self.server = grpc_aio.server()

        self.m_NodeMessageServiceServicer = NodeMessageServiceServicer(self.m_nodeGenerator,self.m_nodeList,self.m_commandLineOptions)

        add_NodeMessageServiceServicer_to_server(
            self.m_NodeMessageServiceServicer, self.server)

        self.server.add_insecure_port('0.0.0.0:8000')
        log(f"Starting HTTP gRPC server on port 8000...")
        if(not (self.m_grpcKeyPEM is None) and not (self.m_grpcCertsPEM is None)):
            server_credentials:grpc.ServerCredentials=grpc.ssl_server_credentials([(self.m_grpcKeyPEM,self.m_grpcCertsPEM)])
            self.server.add_secure_port('0.0.0.0:8001',server_credentials)
            log(f"Starting HTTPS gRPC server on port 8001...")
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
        grpc_thread = threading.Thread(target=self.startGRPCServer, args=(), daemon=True)
        grpc_thread.start()

    def create(self,cmdlineargs:CommandLineOptions)->Flask:
        self.m_commandLineOptions=cmdlineargs
        if(cmdlineargs.useACME()):
            acme_connection:SentraACME=SentraACME(cmdlineargs.getACMEHost(),cmdlineargs.getACMEServerCertificate())
            (self.m_grpcKeyPEM,self.m_grpcCertsPEM)=acme_connection.generateTLSCertsAndKeys()

        self.m_bAppSimulation=cmdlineargs.getRunInSimulationMode()
        self.m_committee_selection=cmdlineargs.getRunCommitteeSelection()
        self.app:Flask = Flask(__name__,static_url_path='',static_folder=self.m_sStaticFolder)
        self.app.add_url_rule("/",view_func=self.getIndex)
        self.app.add_url_rule("/ws", view_func=ws, websocket=True)        
        self.app.add_url_rule("/wsclient", view_func=wsClient, websocket=True)        
        self.app.add_url_rule("/sentra-nodes-table",view_func=self.getIndex)
        self.app.add_url_rule("/sentra-nodes",view_func=self.getIndex)
        self.app.add_url_rule("/api/v1/getNodes",view_func=self.getNodes)
        self.app.add_url_rule("/api/v1/getCommittee",view_func=self.getCommittee)
        self.app.add_url_rule("/api/v1/getPrediction/<int:id>",view_func=self.getPrediction)
        self.app.add_url_rule("/api/v1/postPredictionForPixel",view_func=self.postPredictionForPixel,methods=['POST'])
        self.app.add_url_rule("/api/v1/postReset",view_func=self.postReset,methods=['POST'])
        self.app.add_url_rule("/api/v1/postRemoteAttestation",view_func=self.postRemoteAttestation,methods=['POST'])
        self.app.add_url_rule("/api/v1/postCommitteeSelection",view_func=self.postCommitteeSelection,methods=['POST'])
        self.app.add_url_rule("/api/v1/postRunMPC",view_func=self.postRunMPC,methods=['POST'])
        self.app.add_url_rule("/api/v1/postReceiveResult",view_func=self.postReceiveResult,methods=['POST'])
        self.app.add_url_rule("/api/v1/postReceiveResultOnClient",view_func=self.postReceiveResultOnClient,methods=['POST'])
        self.app.add_url_rule("/api/v1/postNodeSelection/<string:node_id>",view_func=self.postNodeSelection,methods=['POST'])
        self.app.add_url_rule("/api/v1/postImageUpload",view_func=self.postImageUpload,methods=['POST'])

        self.app.add_url_rule("/api/v1/mnist/getTestImagesForDigit/<int:digit>",view_func=self.getMNISTTestImagesForDigit)
        self.app.add_url_rule("/api/v1/mnist/getTestImageForIndex/<int:index>",view_func=self.getMNISTTestImageForIndex)
        self.app.json = CustomJSONProvider(self.app)

        self.m_nodeGenerator=SentraNodeAttributeGenerator(1.0,10.0)
        self.m_nodeList=SentraNodeList()

        if(self.m_bAppSimulation):
            log("Enable Sentra Node Simulation")
            self.m_appSimulator=AppSimulator(self.m_nodeGenerator,self.m_nodeList, self.m_committee_selection)
            self.m_appSimulator.start()

        self.createGRPCServer()

        if (not self.m_bAppSimulation):
            log("distribute secret shares")
            self.m_clientDistributor = ClientDistributor(
                cmdlineargs.training_args)
            for node in self.m_nodeList.m_arNodes:
                self.m_clientDistributor.distribute(node)

        return self.app

    def doReset(self):
        if(self.m_bAppSimulation):
            #Reset is currently only supported for the simulation
            self.m_Committee=None
            self.m_appSimulator.restart()

    def sendMessageToNode(self,node_id:str,message:object)->None:
        self.m_NodeMessageServiceServicer.sendMessageToNode(node_id, message)

    def setCommittee(self,comittee:SentraNodeList):
        self.m_Committee=comittee

    def getIndex(self):
        return send_file(self.m_sIndexHtml)

    #@app.route('/api/v1/getNodes', methods=['GET'])
    def getNodes(self):
        return jsonify(self.m_nodeList)
    
        #@app.route('/api/v1/getCommittee', methods=['GET'])
    def getCommittee(self):
        if(self.m_Committee is None):
            return "{}"
        else:
            return jsonify(self.m_Committee)
        
    #@app.route('/api/v1/postReset', methods=['POST'])
    def postReset(self):
        self.doReset()
        return ""

    #@app.route('/api/v1/postRemoteAttestation', methods=['POST'])
    def postRemoteAttestation(self):
        self.sendWSMessage("{\"doRemoteAttestation\":true}");
        return ""
    
    #@app.route('/api/v1/postCommitteeSelection', methods=['POST'])
    def postCommitteeSelection(self):
        self.sendWSMessage("{\"doCommitteeSelection\":true}");
        print("postCommitteeSelection()",self.m_Committee)
        return jsonify(self.m_Committee)

    #@app.route('/api/v1/postRunMPC', methods=['POST'])
    def postRunMPC(self):
        data=request.get_json()
        bRun:bool=data['doRunMPC']
        if(bRun):
            self.sendWSMessage("{\"doRunMPC\":true}");
        else:
            self.sendWSMessage("{\"doRunMPC\":false}");
        return jsonify({})

    #@app.route('/api/v1/postNodeSelection/<string:node_id>', methods=['POST'])
    def postNodeSelection(self,node_id:str):
        if(self.m_bAppSimulation and self.m_appSimulator):
            node_id=self.m_nodeList.getRandomNode().m_strNodeID
            self.sendWSMessage("{\"doNodeSelection\":\""+node_id+"\"}");
        return ""

    #@app.route('/api/v1/getPrediction/<int:id>', methods=['GET'])    
    def getPrediction(self,id:int):
        if(self.m_bAppSimulation and self.m_appSimulator):
            return jsonify(self.m_appSimulator.doInference(id))
        return jsonify({})

    #@app.route('/api/v1/getPrediction/<int:id>', methods=['GET'])    
    def postPredictionForPixel(self):
        if(self.m_bAppSimulation and self.m_appSimulator):
            data=request.get_json()
            pixels = data['pixels']
            return jsonify(self.m_appSimulator.doInferenceForPixel(pixels))
        return jsonify({})

    #@app.route('/api/v1/postImageUpload', methods=['POST'])    
    def postImageUpload(self):
        data=request.get_json()
        img_b64 = data['image']
        node_id= data['node_id']
        msg:str="{\"imageUpload\":\""+img_b64+"\",\"node_id\":\""+node_id+"\"}"
        self.sendWSMessage(msg)
        return jsonify({})

    #@app.route('/api/v1/postReceiveResult', methods=['POST'])    
    def postReceiveResult(self):
        data=request.get_json()
        img_b64 = data['result']
        node_id= data['node_id']
        msg:str="{\"receiveResult\":\""+img_b64+"\",\"node_id\":\""+node_id+"\"}"
        self.sendWSMessage(msg)
        return jsonify({})

    #@app.route('/api/v1/postReceiveResultOnClient', methods=['POST'])    
    def postReceiveResultOnClient(self):
        data=request.get_json()
        img_b64 = data['result']
        node_id= data['node_id']
        msg:str="{\"receiveResultOnClient\":\""+img_b64+"\",\"node_id\":\""+node_id+"\"}"
        self.sendWSMessageToClient(msg)
        return jsonify({})

    #@app.route("/api/v1/mnist/getTestImagesForDigit/<int:digit>")
    def getMNISTTestImagesForDigit(self,digit:int):
        return jsonify(self.m_MNIST.getJSONObjectForDigit(digit))

    #@app.route("/api/v1/mnist/getTestImageForIndex/<int:index>")
    def getMNISTTestImageForIndex(self,index:int):
        return jsonify(self.m_MNIST.getJSONObjectForTestImage(index))

    def registerWebSocket(self, ws:Server):
        if(self.m_WS is not None and self.m_WS.connected):
            try:
                self.m_WS.close()
            except:
                pass
        self.m_WS = ws

    def registerClientWebSocket(self, ws:Server):
        if(self.m_WSClient is not None and self.m_WSClient.connected):
            try:
                self.m_WSClient.close()
            except:
                pass
        self.m_WSClient = ws

    def sendWSMessage(self,msg:str):
        if(not self.m_WS is None):
            self.m_lockWS.acquire()
            if(self.m_WS.connected):
                try:
                    self.m_WS.send(msg)
                except:
                    try:
                        self.m_WS.close()
                    except:
                        pass
                    self.m_WS=None
            else:
                self.m_WS=None
            self.m_lockWS.release()
    
    def sendWSMessageToClient(self,msg:str):
        if(not self.m_WSClient is None):
            self.m_lockWSClient.acquire()
            if(self.m_WSClient.connected):
                try:
                    self.m_WSClient.send(msg)
                except:
                    try:
                        self.m_WSClient.close()
                    except:
                        pass
                    self.m_lockWSClient=None
            else:
                self.m_lockWSClient=None
            self.m_lockWSClient.release()

    def notifyNodeListUpdated(self):
        data:object={"cloudUpdate":
                {
                  "nodeList":self.m_nodeList,
                  "committee":self.m_Committee
                }
              }
        self.sendWSMessage(self.app.json.dumps(data))

# @app.route("/ws",websocket=True)
def ws():
    backend:Backend|None=Backend.getBackend()
    if(backend is None):
        return "",500
    try:
       
        if(backend.m_WS is not None and backend.m_WS.connected):
            return "",200
        ws = Server.accept(request.environ)
        
        backend.registerWebSocket(ws)
        while ws.connected:
            ws.receive()
        backend.m_WS=None
    except:
        backend.m_WS=None
        return "",500
    return "",200

# @app.route("/wsclient",websocket=True)
def wsClient():
    backend:Backend|None=Backend.getBackend()
    if(backend is None):
        return "",500
    try:
       
        if(backend.m_WSClient is not None and backend.m_WSClient.connected):
            return "",200
        ws = Server.accept(request.environ)
        
        backend.registerClientWebSocket(ws)
        while ws.connected:
            ws.receive()
        backend.m_WSClient=None
    except:
        backend.m_WSClient=None
        return "",500
    return "",200
