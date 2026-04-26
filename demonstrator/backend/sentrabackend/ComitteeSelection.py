import time
from collections import defaultdict

from .SentraNodeList import SentraNodeList
from .SentraNode import SentraNode
from .Log import log as log

class CommitteeSelection:
    """
    Implements the committee selection algorithm of Sentra.

    Attributes:
        m_old_committee (SentraNodeList): The committee of the previous epoch,
        if exists.
        m_target_size (int): Target size of the committee.
        m_min_trust (float): Required minimal trust score of a node to be
        considered as candidate.
        m_max_attest_age (int): Maximum time after last attestation.
        m_max_hw_frac (float): Diversity constraint 1 - maximum fraction of
        nodes in the committee running on the same hardware platform.
        m_max_op_frac (float): Diversity constraint 2 - maximum fraction of
        nodes in the committee run by the same operator.
        m_max_pm_frac (float): Diversity constraint 3 - maximum fraction of
        nodes in the committee running on the same CPU.
    """

    m_old_committee: SentraNodeList = SentraNodeList()
    m_target_size: int
    m_min_trust: float
    m_max_attest_age: int
    m_max_hw_frac: float
    m_max_op_frac: float
    m_max_pm_frac: float

    def __init__(self, target_size: int, min_trust: int, max_attest_age: int,
                 max_hw_frac: float, max_op_frac: float, max_pm_frac: float):
        """
        Initializes the CommitteeSelection.

        Args:
            target_size (int): Value stored in the node.
            min_trust (int): Minimal trust score of a node to be considered as
            candidate.
            max_attest_age (int): Maximum time after last attestation.
            max_hw_frac (int): Maximum fraction of nodes in the committee
            running on the same hardware platform.
            max_op_frac (int): Maximum fraction of nodes in the committee run
            by the same operator.
            max_pm_frac (int): Maximum fraction of nodes in the committee
            running on the same CPU.
        """

        if not all(0 <= x <= 1 for x in (max_hw_frac, max_op_frac, max_pm_frac)):
            raise ValueError(
                "max_hw_frac, max_op_frac, max_pm_frac must be in range 0-1")

        if not all(x > 0 for x in (target_size, min_trust, max_attest_age)):
            raise ValueError(
                "target_size, min_trust, max_attest_age must be \
                greater then zero")

        self.m_target_size = target_size
        self.m_min_trust = min_trust
        self.m_max_attest_age = max_attest_age
        self.m_max_hw_frac = max_hw_frac
        self.m_max_op_frac = max_op_frac
        self.m_max_pm_frac = max_pm_frac

    def filter(self, nodes: SentraNodeList) -> SentraNodeList:
        """
        Step 1 of node selection algorithm.

        Filter the candidate list for all nodes which fulfill the attestation
        requirements.

        Args:
            nodes (SentraNodeList): Available nodes.

        Returns:
            SentraNodeList: Candidate nodes.
        """

        curr_time = time.time()
        candidates:SentraNodeList=SentraNodeList()

        for node in list(nodes.m_arNodes.values()):
            if not node.m_bVerified:
                log(f"removing node {node.m_strNodeID} from candidate set, \
                     reason: not verified")
                continue

            if node.m_fTrustScore < self.m_min_trust:
                log(f"removing node {node.m_strNodeID} from candidate set, \
                     reason: trustScore too low")
                continue

            if curr_time - node.m_attestTime > self.m_max_attest_age:
                log(f"removing node {node.m_strNodeID} from candidate set, \
                    reason: attestation too old")
                continue
            candidates.add(node)

        return candidates

    def check_reuse(self):
        """
        Step 2 of node selection algorithm.

        Check, if the previous committee (if existent) can be reused.

        Returns:
            Bool: If old committee can be reused.
        """

        if all(node_id in self.m_candidates.m_arNodes
               for node_id in self.m_old_committee.m_arNodes):

            log("previous committee can be reused")
            return True

        else:

            log("previous committee can't be reused")
            return False

    # TODO: so far only returns the existing value.
    def score(self, node: SentraNode):
        """
        Step 3 of node selection algorithm.

        Compute the score of a single node.

        Args:
            node (SentraNode): Node to compute the trust score of.

        Returns:
            float: Trust score of the node.
        """

        return node.m_fTrustScore

    def greedy_select(self, sorted_candidates:list[SentraNode])->SentraNodeList:
        """
        Step 4 of node selection algorithm.

        Select the committee for epoch e+1 from the sorted candidate list using a greedy algorithm. Selection is based on the configured diversity constraints.

        Args:
            sorted_candidates (SentraNodeList): Sorted list of candidate nodes.

        Returns:
            SentraNodeList: Node list of new committee. If length == 0, no committee was found.
        """

        count_hw:defaultdict[int,int] = defaultdict(int)
        count_op:defaultdict[int,int] = defaultdict(int)
        count_pm:defaultdict[int,int] = defaultdict(int)

        new_committee = SentraNodeList()

        for node in sorted_candidates:
            node_hw = node.m_iCPUArchitecture
            node_op = node.m_iOperator
            node_pm = node.m_iHost
            frac_hw_new = (count_hw.get(node_hw, 0) + 1) / self.m_target_size
            frac_op_new = (count_op.get(node_op, 0) + 1) / self.m_target_size
            frac_pm_new = (count_pm.get(node_pm, 0) + 1) / self.m_target_size

            if (
                frac_hw_new <= self.m_max_hw_frac
                and frac_op_new <= self.m_max_op_frac
                and frac_pm_new <= self.m_max_pm_frac
            ):

                count_hw[node_hw] = count_hw.get(node_hw, 0) + 1
                count_op[node_op] = count_op.get(node_op, 0) + 1
                count_pm[node_pm] = count_pm.get(node_pm, 0) + 1
                new_committee.add(node)

                log(f"node {node.m_strNodeID} fulfills requirements")

                if new_committee.len() == self.m_target_size:
                    return new_committee

        return new_committee

    def selectionAlgorithm(self, nodes: SentraNodeList) -> SentraNodeList|None:
        """
        Node selection algorithm.

        Select the committee for epoch e+1 from a list of available nodes.

        Args:
            nodes (SentraNodeList): List of available nodes.

        Returns:
            SentraNodeList: Node list of new committee. If length == 0,
            no committee was found.
        """

        log("select nodes for committee")
        self.m_candidates = self.filter(nodes)

        new_committee: SentraNodeList = SentraNodeList()

        if self.m_candidates.len() < self.m_target_size:
            log(f"candidate list size after filtering: \
                {self.m_candidates.len()} is smaller than required \
                committee size: {self.m_target_size}")
            return None

        if self.m_old_committee.len():
            if (self.check_reuse()):
                return self.m_old_committee

        sorted_candidates = sorted(self.m_candidates.m_arNodes.values(),
                                   key=lambda node: node.m_fTrustScore)

        new_committee = self.greedy_select(sorted_candidates)

        if not new_committee.len():
            log("no committee found")
            return new_committee

        else:
            log(f"found committee of size {new_committee.len()}")

        self.m_old_committee = new_committee

        return new_committee
