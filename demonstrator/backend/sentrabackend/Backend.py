from flask import Flask, jsonify
from flask import send_file
import threading
import asyncio
import grpc.aio as grpc_aio

from .CommandLineOptions import CommandLineOptions
from .CustomJSONProvider import CustomJSONProvider
from .SentraNodeAttributeGenerator import SentraNodeAttributeGenerator
from .SentraNodeList import SentraNodeList
from .AppSimulator import AppSimulator
from .grpc import NodeMessageServiceServicer
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

    def __init__(self):
        self.m_bAppSimulation=False
        self.m_appSimulator=None

    async def runGRPCServer(self):
        self.server = grpc_aio.server()
        self.m_NodeMessageServiceServicer = NodeMessageServiceServicer(self.m_nodeGenerator,self.m_nodeList)
        add_NodeMessageServiceServicer_to_server(self.m_NodeMessageServiceServicer,
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