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

class Backend:

    m_sStaticFolder="../frontend/dist/frontend/browser"
    m_sIndexHtml=m_sStaticFolder+"/index.html"
    m_nodeGenerator:SentraNodeAttributeGenerator
    m_nodeList:SentraNodeList
    m_bAppSimulation:bool=False
    m_appSimulator:AppSimulator|None=None
    m_NodeMessageServiceServicer:NodeMessageServiceServicer
    m_grpcCertsPEM:bytes|None
    m_grpcKeyPEM:bytes|None

    def __init__(self):
        self.m_bAppSimulation=False
        self.m_appSimulator=None
        self.m_grpcCertsPEM=None
        self.m_grpcKeyPEM=None

    async def runGRPCServer(self):
        self.server = grpc_aio.server()
        self.m_NodeMessageServiceServicer = NodeMessageServiceServicer(self.m_nodeGenerator,self.m_nodeList)
        add_NodeMessageServiceServicer_to_server(self.m_NodeMessageServiceServicer,self.server)
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
#        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        grpc_thread = threading.Thread(target=self.startGRPCServer, args=(), daemon=True)
        grpc_thread.start()

    def create(self,cmdlineargs:CommandLineOptions)->Flask:
        if(cmdlineargs.useACME()):
            acme_connection:SentraACME=SentraACME(cmdlineargs.getACMEHost(),cmdlineargs.getACMEServerCertificate())
            (self.m_grpcKeyPEM,self.m_grpcCertsPEM)=acme_connection.generateTLSCertsAndKeys()

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