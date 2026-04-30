import { ChangeDetectorRef, Component, ViewChild } from '@angular/core';
import { SentraNodeGraph } from '../sentra-nodes-graph/sentra-nodes-graph';
import { RestService, SentraNode } from '../../rest.service';
import { App } from '../app';
import { createWebSocketURLForPath, sleep } from '../../utils';
import { Router } from '@angular/router';

@Component({
  selector: 'app-sentra-cloud-dashboard',
  imports: [SentraNodeGraph],
  templateUrl: './sentra-cloud-dashboard.html',
  styleUrl: './sentra-cloud-dashboard.css',
})


export class SentraCloudDashboard {
@ViewChild('graph') graph!: SentraNodeGraph;
  m_socket?:WebSocket=undefined;

  m_NodeList?:SentraNode[]=undefined;
  m_Committee?:SentraNode[]=undefined;

constructor(private m_RestService: RestService,private cdr: ChangeDetectorRef,private router: Router)
  {
    this.receiveWebSocketMsg();
    this.autoWebSocketReconnect();

    App.staticSubTitle="Cloud Side";
    this.m_RestService.getNodes().subscribe(async (nodes)=>
      {
        this.m_NodeList=nodes;
      console.log("Loaded nodes: "+nodes.length);
      for (let node of nodes)
      {
        this.graph.add(node.node_id,node.label,node.host,node.cpu,node.operator);
      }
      this.graph.layout();
      this.m_RestService.getCommittee().subscribe(async (committee)=>
      {
        console.log("Loaded committee: "+committee.length);
        this.m_Committee=committee;


      });
    });
  }

  async showAttestedNodes()
  {
    if(this.m_NodeList===undefined)
    {
      return;
    }
 for (let node of this.m_NodeList)
      {
        this.graph.setAttested(node.node_id,node.attested);
        this.graph.stopPulseBorder(node.node_id);
        await sleep(2000);
      }
       
  }

  async showCommittee()
  {
if(this.m_Committee===undefined)
  return;
 let i:number=0;
  for (i=0;i<this.m_Committee.length;i++)
          {
            let c=this.m_Committee[i];
            this.graph.setCommitteeMember(c.node_id,true);
            let j:number=0;
            let nodeid1:string=c.node_id;
            for(j=i+1;j<this.m_Committee.length;j++)
            {
              this.graph.addEdge(nodeid1,this.m_Committee[j].node_id);
            }
            await sleep(2000);
          }    
  }

  handleCloudUpdate(cloudUpdate: any) {
 /*   this.graph.clear();
    for (let node of cloudUpdate.nodeList)
      {
        this.graph.add(node.node_id,node.label,
          node.host,node.cpu,node.operator
        )
        this.graph.setAttested(node.node_id,node.attested);
      }
     this.graph.layout();
     for (let c of cloudUpdate.committee)
          {
            this.graph.setCommitteeMember(c.node_id,true);
          }
//    this.cdr.detectChanges();    
//      this.router.navigate([this.router.url]);
 */         window.location.reload();
  }

  handleRemoteAttestation(remoteAttestation:any)
  {
    this.showAttestedNodes();
  }

    handleCommitteeSelection(committeeSelection:any)
  {
    this.showCommittee();
  }


  //do something with the web socket
receiveWebSocketMsg()
{
  if(this.m_socket&&(this.m_socket.readyState === WebSocket.OPEN || this.m_socket.readyState === WebSocket.CONNECTING))
    return; //do nothing if already connected
  this.m_socket= new WebSocket(createWebSocketURLForPath("/ws"));
  this.m_socket.onerror=((ev:any)=>
    {
    }
  );
  this.m_socket.onclose=((ev:any)=>
    {
    }
  );

  this.m_socket.onmessage=((ev:any)=>
  {
    const obj = JSON.parse(ev.data);
   if('cloudUpdate' in obj) //Cloud has changed
    {
      this.handleCloudUpdate(obj.cloudUpdate);
    }
    else if('doRemoteAttestation' in obj)
    {
      this.handleRemoteAttestation(obj.remoteAttestation);
    }
    else if('doCommitteeSelection' in obj)
    {
      this.handleCommitteeSelection(obj.committeeSelection);
    }
    console.log(ev.data)
  });
}

autoWebSocketReconnect()
{
  setInterval(()=>this.receiveWebSocketMsg(),10000)
}



}
