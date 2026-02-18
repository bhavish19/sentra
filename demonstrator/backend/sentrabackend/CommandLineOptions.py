import argparse

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
