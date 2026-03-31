import threading
import random
import time

from .SentraNodeAttributeGenerator import SentraNodeAttributeGenerator
from .SentraNodeList import SentraNodeList
from .SentraNode import SentraNode
from .ComitteeSelection import CommitteeSelection
from .Log import log as log

class AppSimulator:

    m_Thread:threading.Thread|None
    m_nodeGenerator:SentraNodeAttributeGenerator

    def __init__(self,nodeGenerator:SentraNodeAttributeGenerator,nodeList:SentraNodeList, committeeSelection:bool):
        self.m_nodeGenerator=nodeGenerator
        self.m_nodeList=nodeList
        self.m_committeeSelection = committeeSelection

    def runSimulation(self):
        i:int=0
        baseId:str="SentraNode_"
        while(i<10):
            node:SentraNode=self.m_nodeGenerator.generateNode(baseId+str(i))
            if(random.random()>0.1):
                node.setVerified(True, time.time())
            self.m_nodeList.add(node)
            i+=1

        if self.m_committeeSelection:
            log("node list:")
            log(str(self.m_nodeList))

            committee_target_size = 5
            committee_min_trust = 2
            committee_max_attest_age = 100
            committee_max_hw_frac = 0.8
            committee_max_op_frac = 0.8
            committee_max_pm_frac = 0.8

            committeeSelection: CommitteeSelection = CommitteeSelection(committee_target_size, committee_min_trust, committee_max_attest_age, committee_max_hw_frac, committee_max_op_frac, committee_max_pm_frac)

            committee = committeeSelection.selectionAlgorithm(self.m_nodeList)

            log(f"running committee selction with: committee target size: {committee_target_size}, committee min trust: {committee_min_trust}, committee_max_attest_age: {committee_max_attest_age}, committee_max_hw_frac: {committee_max_hw_frac}, committee_max_op_frac: {committee_max_op_frac}, committee_max_pm_frac: {committee_max_pm_frac}")

            if committee:
                log("committee:")
                log(str(committee))
            else:
                log("no committee found!")

    def start(self):
        self.m_Thread = threading.Thread(target=self.runSimulation, args=(), daemon=True)
        self.m_Thread.start()