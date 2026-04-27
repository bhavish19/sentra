from flask import Flask, jsonify
from flask import send_file
import threading
import asyncio
import grpc
import grpc.aio as grpc_aio

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
    def getBackend(cls) -> "Backend":
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
        self.app.add_url_rule("/sentra-nodes-table",view_func=self.getIndex)
        self.app.add_url_rule("/sentra-nodes",view_func=self.getIndex)
        self.app.add_url_rule("/api/v1/getNodes",view_func=self.getNodes)
        self.app.add_url_rule("/api/v1/getCommittee",view_func=self.getCommittee)
        self.app.add_url_rule("/api/v1/getPrediction/<int:id>",view_func=self.getPrediction)
        self.app.add_url_rule("/api/v1/postReset",view_func=self.postReset,methods=['POST'])
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
    
    #@app.route('/api/v1/getPrediction/<int:id>', methods=['GET'])    
    def getPrediction(self,id:int):
        if(self.m_bAppSimulation and self.m_appSimulator):
            return jsonify(self.m_appSimulator.doInference(id))
        return jsonify({})
    
    #@app.route("/api/v1/mnist/getTestImagesForDigit/<int:digit>")
    def getMNISTTestImagesForDigit(self,digit:int):
        return jsonify(self.m_MNIST.getJSONObjectForDigit(digit))

    #@app.route("/api/v1/mnist/getTestImageForIndex/<int:index>")
    def getMNISTTestImageForIndex(self,index:int):
        return jsonify(self.m_MNIST.getJSONObjectForTestImage(index))


    