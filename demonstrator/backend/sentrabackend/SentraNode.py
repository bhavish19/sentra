from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .SentraNodeAttributeGenerator import SentraNodeAttributeGenerator

class SentraNode:
    m_strNodeID:str
    m_iCPUArchitecture:int
    m_iOperator:int
    m_iHost:int
    m_fTrustScore:float
    m_attestTime: float
    m_bVerified:bool
    m_sendQueue: asyncio.Queue[object]
    m_nodeGenerator:SentraNodeAttributeGenerator

    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeID:str,trustscore:float,cpu:int,host:int,operator:int):
        self.m_iOperator=operator
        self.m_fTrustScore=trustscore
        self.m_iCPUArchitecture=cpu
        self.m_iHost=host
        self.m_strNodeID=nodeID
        self.m_bVerified=False
        self.m_attestTime=0
        self.m_sendQueue= asyncio.Queue()
        self.m_nodeGenerator=nodeGenerator

    def __str__(self)->str:
        ret=f"SentraNode {self.m_strNodeID}:\n"
        ret+=f"\tOperator:    {self.getOperatorName()}\n"
        ret+=f"\tHost:        {self.getHostName()}\n"
        ret+=f"\tCPU:         {self.getCPUName()}\n"
        ret+=f"\tattested:    {self.m_bVerified} (at: {self.m_attestTime}\n"
        ret+=f"\tTrust score: {self.m_fTrustScore}\n"
        return ret

    def __hash__(self):
        return hash(self.m_strNodeID)

    def __eq__(self, other:object)->bool:
        if not isinstance(other, SentraNode):
            return NotImplemented
        return self.m_strNodeID == other.m_strNodeID

    def setVerified(self,b:bool,time:float)->None:
        self.m_bVerified=b
        self.m_attestTime=time

    def getSendQueue(self)-> asyncio.Queue[object]:
        return self.m_sendQueue

    def getOperatorName(self)->str:
        return self.m_nodeGenerator.getOpertor(self.m_iOperator)

    def getHostName(self)->str:
        return self.m_nodeGenerator.getHost(self.m_iHost).m_Name

    def getCPUName(self)->str:
        return self.m_nodeGenerator.getCPUForHost(self.m_iHost)

    def toJSONObject(self)->object:
            return {
                'node_id':self.m_strNodeID,
                'host':self.getHostName(),
                'operator':self.getOperatorName(),
                'cpu':self.getCPUName(),
                'attested':self.m_bVerified
            }
