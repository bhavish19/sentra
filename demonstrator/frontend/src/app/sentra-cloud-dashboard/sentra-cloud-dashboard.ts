import { Component, ViewChild } from '@angular/core';
import { SentraNodeGraph } from '../sentra-nodes-graph/sentra-nodes-graph';
import { RestService, SentraNode } from '../../rest.service';

@Component({
  selector: 'app-sentra-cloud-dashboard',
  imports: [SentraNodeGraph],
  templateUrl: './sentra-cloud-dashboard.html',
  styleUrl: './sentra-cloud-dashboard.css',
})


export class SentraCloudDashboard {
@ViewChild('graph') graph!: SentraNodeGraph;

constructor(private m_RestService: RestService)
  {
    this.m_RestService.getNodes().subscribe((nodes)=>
      {
      console.log("Loaded nodes: "+nodes.length);
      for (let node of nodes)
      {
        this.graph.add(node.node_id,node.label,
          node.host,node.cpu,node.operator
        )
        this.graph.setAttested(node.node_id,node.attested);
      }
      this.graph.layout();
      this.m_RestService.getCommittee().subscribe((committee)=>
      {
        console.log("Loaded committee: "+committee.length);
        for (let c of committee)
          {
            this.graph.setCommitteeMember(c.node_id,true);
          }
      });
    });
  }
}
