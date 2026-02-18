import time
import random

from .Host import Host
from .SentraNode import SentraNode
 
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
    