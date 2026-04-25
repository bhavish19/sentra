import { Component, ViewChild, AfterViewInit, ChangeDetectionStrategy } from '@angular/core';
import { signal } from '@angular/core'; // Angular 16+ signals
import { CytoscapeGraphComponent } from 'cytoscape-angular';

// --- Node and Edge Types ---
type NodeType = 'cloud' | 'server' | 'process';
type Architecture = 'ARM' | 'Intel' | 'AMD';
type ProcessStatus = 'running' | 'stopped' | 'error' | 'unknown';

interface CloudNode {
  data: { id: string; type: 'cloud'; label: string; icon: string; };
}
interface ServerNode {
  data: { id: string; type: 'server'; label: string; parent: string; arch: Architecture; icon: string; };
}
interface ProcessNode {
  data: {
    id: string;
    type: 'process';
    label: string;
    parent: string;
    icon: string;
    status?: ProcessStatus;
    statusColor?: string;
  };
}
type GraphNode = CloudNode | ServerNode | ProcessNode;

interface GraphEdge {
  data: { id: string; source: string; target: string; };
}

@Component({
  selector: 'app-cloud-infra-graph',
  templateUrl: './cloud-infra-graph.component.html',
  styleUrls: ['./cloud-infra-graph.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class CloudInfraGraphComponent implements AfterViewInit {
  @ViewChild(CytoscapeGraphComponent) cyGraph!: CytoscapeGraphComponent;

  // --- Reactive graph state ---
  nodes = signal<GraphNode[]>([]);
  edges = signal<GraphEdge[]>([]);

  // --- Icon URLs (replace with your own asset paths) ---
  private cloudIcons: Record<string, string> = {
    'AWS': 'assets/icons/cloud-aws.svg',
    'Azure': 'assets/icons/cloud-azure.svg',
    'GCP': 'assets/icons/cloud-gcp.svg',
    'default': 'assets/icons/cloud-generic.svg'
  };
  private serverIcons: Record<Architecture, string> = {
    'ARM': 'assets/icons/server-arm.svg',
    'Intel': 'assets/icons/server-intel.svg',
    'AMD': 'assets/icons/server-amd.svg'
  };
  private processIcon = 'assets/icons/process.svg';

  // --- Status color mapping for pie overlays ---
  private statusColors: Record<ProcessStatus, string> = {
    'running': '#4caf50',
    'stopped': '#f44336',
    'error': '#ff9800',
    'unknown': '#bdbdbd'
  };

  // --- Cytoscape style ---
  style = [
    // Cloud nodes
    {
      selector: 'node[type="cloud"]',
      style: {
        'background-image': 'data(icon)',
        'background-fit': 'cover',
        'shape': 'ellipse',
        'width': 120,
        'height': 80,
        'label': 'data(label)',
        'font-size': 18,
        'text-valign': 'bottom',
        'text-halign': 'center'
      }
    },
    // Server nodes
    {
      selector: 'node[type="server"]',
      style: {
        'background-image': 'data(icon)',
        'background-fit': 'cover',
        'shape': 'round-rectangle',
        'width': 80,
        'height': 50,
        'label': 'data(label)',
        'font-size': 14,
        'text-valign': 'bottom',
        'text-halign': 'center'
      }
    },
    // Process nodes (with status pie overlay)
    {
      selector: 'node[type="process"]',
      style: {
        'background-image': 'data(icon)',
        'background-fit': 'cover',
        'shape': 'ellipse',
        'width': 40,
        'height': 40,
        'label': 'data(label)',
        'font-size': 12,
        'text-valign': 'bottom',
        'text-halign': 'center',
        // Pie overlay for status
        'pie-size': '20%',
        'pie-1-background-color': 'data(statusColor)',
        'pie-1-background-size': 25
      }
    },
    // Edges (optional, for visual clarity)
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#bdbdbd',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#bdbdbd',
        'curve-style': 'bezier'
      }
    }
  ];

  // --- Layout options ---
  layoutOptions = {
    name: 'cose-bilkent',
    animate: true,
    fit: true,
    padding: 30
  };

  // --- Public API: Add a process (auto-creates cloud/server if needed) ---
  /**
   * Adds a process to the graph, auto-creating cloud/server if missing.
   * @param processName Name of the process
   * @param serverId ID of the server
   * @param architecture Server architecture ('ARM' | 'Intel' | 'AMD')
   * @param cloud Name of the cloud (e.g., 'AWS')
   * @param status Optional process status ('running', 'stopped', 'error', 'unknown')
   */
  add(
    processName: string,
    serverId: string,
    architecture: Architecture,
    cloud: string,
    status: ProcessStatus = 'unknown'
  ) {
    // --- Ensure cloud node exists ---
    let nodesArr = this.nodes();
    let cloudId = `cloud-${cloud}`;
    if (!nodesArr.find(n => n.data.id === cloudId)) {
      nodesArr = [
        ...nodesArr,
        {
          data: {
            id: cloudId,
            type: 'cloud',
            label: cloud,
            icon: this.cloudIcons[cloud] || this.cloudIcons['default']
          }
        }
      ];
    }

    // --- Ensure server node exists ---
    let serverNodeId = `server-${serverId}`;
    if (!nodesArr.find(n => n.data.id === serverNodeId)) {
      nodesArr = [
        ...nodesArr,
        {
          data: {
            id: serverNodeId,
            type: 'server',
            label: serverId,
            parent: cloudId,
            arch: architecture,
            icon: this.serverIcons[architecture]
          }
        }
      ];
    }

    // --- Add process node ---
    let processNodeId = `process-${serverId}-${processName}`;
    if (!nodesArr.find(n => n.data.id === processNodeId)) {
      nodesArr = [
        ...nodesArr,
        {
          data: {
            id: processNodeId,
            type: 'process',
            label: processName,
            parent: serverNodeId,
            icon: this.processIcon,
            status,
            statusColor: this.statusColors[status]
          }
        }
      ];
    }

    // --- Optionally, add edges for clarity (not required for compound nodes) ---
    let edgesArr = this.edges();
    // Edge: server -> process
    if (!edgesArr.find(e => e.data.id === `edge-${serverNodeId}-${processNodeId}`)) {
      edgesArr = [
        ...edgesArr,
        { data: { id: `edge-${serverNodeId}-${processNodeId}`, source: serverNodeId, target: processNodeId } }
      ];
    }
    // Edge: cloud -> server
    if (!edgesArr.find(e => e.data.id === `edge-${cloudId}-${serverNodeId}`)) {
      edgesArr = [
        ...edgesArr,
        { data: { id: `edge-${cloudId}-${serverNodeId}`, source: cloudId, target: serverNodeId } }
      ];
    }

    // --- Update signals (triggers Cytoscape re-render) ---
    this.nodes.set(nodesArr);
    this.edges.set(edgesArr);
  }

  ngAfterViewInit() {
    // Optionally, access Cytoscape core instance if needed
    // this.cyGraph.getCytoscapeInstance();
  }
}
