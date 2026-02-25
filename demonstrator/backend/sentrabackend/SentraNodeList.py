import threading
import asyncio

from .SentraNode import SentraNode

class SentraNodeList:
    m_arNodes:dict[str,SentraNode]={}
    m_Lock:threading.Lock

    def __init__(self):
        self.m_arNodes={}
        self.m_Lock=threading.Lock()

    def __str__(self)->str:
        ret=f"SentraNode list with {self.len()} nodes:\n"
        for node in self.m_arNodes.values():
            ret+=str(node)
        return ret

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

    def setVerified(self,node_id:str,time:float)->bool:
        with self.m_Lock:
            node:SentraNode|None=self.m_arNodes.get(node_id,None)
            if(not node is None):
                node.setVerified(True,time)
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
    
    def getNodes(self)->list[SentraNode]:
        with self.m_Lock:
            return self.m_arNodes.values()
