import asyncio

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
