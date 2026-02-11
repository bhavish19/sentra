import argparse
from flask import Flask, jsonify, request
from flask import send_file

import grpc
import threading
import SentraBackend_GRPC_Services_pb2
import SentraBackend_GRPC_Services_pb2_grpc
from concurrent import futures

BACKEND_VERSION="00.01.001"

class CommandLineOptions:

    m_Parser=argparse.ArgumentParser("Sentra Backend",
                                     "This programm is the Sentra backend. It manages the Sentra nodes and the related training/inference process.",
                                     "Below are the valid command line options.")

    def __init__(self):
        self.m_Parser.add_argument("-p","--port",help="Port to listen on for Web/REST-API requests",default=8888)
        self.m_Parser.add_argument("-l","--host",help="Host to listen on for Web/REST-API requests",default="127.0.0.1")
        self.m_Args=self.m_Parser.parse_args()
        
    def getPort(self)->int:
        return self.m_Args.port

    def getHost(self)->str:
        return self.m_Args.host
        
class NodeRegistrationServicer(SentraBackend_GRPC_Services_pb2_grpc.NodeRegistrationServicer):
    
    def RegisterNode(self, request, context):
        node_id = request.node_id
        
        if not node_id:
            return SentraBackend_GRPC_Services_pb2.RegisterResponse(
                success=False,
                message="Node ID cannot be empty"
            )
                
        print(f"Node registered: {node_id}")
        
        return SentraBackend_GRPC_Services_pb2.RegisterResponse(
            success=True,
            message=f"Node {node_id} registered successfully"
        )

class Backend:

    m_sStaticFolder="../frontend/dist/frontend/browser"
    m_sIndexHtml=m_sStaticFolder+"/index.html"

    def __init__(self):
        pass
    

    def runGRPCServer(self):
        self.server.wait_for_termination()

    def createGRPCServer(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        servicer = NodeRegistrationServicer()
        SentraBackend_GRPC_Services_pb2_grpc.add_NodeRegistrationServicer_to_server(servicer, self.server)
        self.server.add_insecure_port('127.0.0.1:8000')    
        print(f"Starting gRPC server on port 8000...")
        self.server.start()
        grpc_thread = threading.Thread(target=self.runGRPCServer, args=(), daemon=True)
        grpc_thread.start()


    def create(self,cmdlineargs:CommandLineOptions)->Flask:
        self.app:Flask = Flask(__name__,static_url_path='',static_folder=self.m_sStaticFolder)
        self.app.add_url_rule("/",view_func=self.getIndex)

        self.createGRPCServer()
        return self.app

    def getIndex(self):
        return send_file(self.m_sIndexHtml)

if __name__ == '__main__':
    print("Starting Sentra Backend...")
    print("Version: ",BACKEND_VERSION)
    cmdlineargs=CommandLineOptions()
    backend=Backend()
    app:Flask=backend.create(cmdlineargs)
    app.run(debug=False,port=cmdlineargs.getPort(),host=cmdlineargs.getHost(),threaded=True)
