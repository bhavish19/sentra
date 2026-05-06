import { ChangeDetectorRef, Component, ViewChild } from '@angular/core';
import { SentraNodeGraph } from '../sentra-nodes-graph/sentra-nodes-graph';
import { RestService, SentraNode } from '../../rest.service';
import { App } from '../app';
import { createWebSocketURLForPath, sleep } from '../../utils';
import { Router } from '@angular/router';
import { Position } from 'cytoscape';
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
  m_strSelectedNodeID:string|null=null;

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
        this.graph.pulseBorder(node.node_id);
      }

    for (let node of this.m_NodeList)
      {
        await sleep(2000);
        this.graph.setAttested(node.node_id,node.attested);
        this.graph.stopPulseBorder(node.node_id);
      }
       
  }

    nodeSelection(node_id:string)
  {
        console.log("Select node: "+node_id);
        this.m_strSelectedNodeID=node_id;
        this.graph.pulseBorder(node_id);
       
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

  async doRunMPC(bRunMPC:boolean)
  {
if(this.m_Committee===undefined)
  return;
 let i:number=0;
  for (i=0;i<this.m_Committee.length;i++)
          {
            let c=this.m_Committee[i];
            let j:number=0;
            let nodeid1:string=c.node_id;
            for(j=i+1;j<this.m_Committee.length;j++)
            {
              this.graph.animateCircle(nodeid1,this.m_Committee[j].node_id,bRunMPC);
            }
            await sleep(50);
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

  handleNodeSelection(nodeSelection:any)
  {
    let node_id:string=String(nodeSelection)
    this.nodeSelection(node_id);
     
  }

  handleImageUpload(imgMsg:any)
  {
    console.log("handleImageUpload() - recevied message: ",imgMsg);
    let img=imgMsg.imageUpload;
    let node_id=imgMsg.node_id;
    console.log("handleImageUpload() - recevied message for node: ",node_id);
    if(node_id==="")
    {      
      if(this.m_strSelectedNodeID===null)
      {
        return;
      }
      else
      {
        node_id=this.m_strSelectedNodeID;
      }
    }
    this.graph.doImageUpload(img,node_id);

  }


doReceiveResult(resultMsg:any)
{
    console.log("doReceiveResult() - recevied message: ",resultMsg);
    let img=resultMsg.receiveResult;
    let node_id=resultMsg.node_id;
    let bPlain:boolean=false;
    console.log("doReceiveResult() - received message for node: ",node_id);
    if(node_id==="")
    {      
      if(this.m_strSelectedNodeID===null)
      {
        return;
      }
      else
      {
        node_id=this.m_strSelectedNodeID;
        bPlain=true;
      }
    }
    this.graph.doResultDownload(img,node_id,bPlain);

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
      this.handleRemoteAttestation(obj.doRemoteAttestation);
    }
    else if('doCommitteeSelection' in obj)
    {
      this.handleCommitteeSelection(obj.doCommitteeSelection);
    }
    else if('doRunMPC' in obj)
    {
      this.doRunMPC(obj.doRunMPC);
    }
    else if('receiveResult' in obj)
    {
      this.doReceiveResult(obj);
    }

    else if('doNodeSelection' in obj)
    {
      this.handleNodeSelection(obj.doNodeSelection);
    }
    else if('imageUpload' in obj)
    {
      this.handleImageUpload(obj);
    }
    console.log(ev.data)
  });
}

autoWebSocketReconnect()
{
  setInterval(()=>this.receiveWebSocketMsg(),10000)
}



}
