import { ChangeDetectorRef, Component, ViewChild } from '@angular/core';
import { SentraNodeGraph } from '../sentra-nodes-graph/sentra-nodes-graph';
import { RestService, SentraNode } from '../../rest.service';
import { App } from '../app';
import { createWebSocketURLForPath } from '../../utils';
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

constructor(private m_RestService: RestService,private cdr: ChangeDetectorRef,private router: Router)
  {
    this.receiveWebSocketMsg();
    this.autoWebSocketReconnect();

    App.staticSubTitle="Cloud Side";
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
    console.log(ev.data)
  });
}

autoWebSocketReconnect()
{
  setInterval(()=>this.receiveWebSocketMsg(),10000)
}



}
