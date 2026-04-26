/**
 * TopologyGraphComponent
 * ----------------------
 * Angular 17+ standalone component for visualizing cloud/server/process topologies using Cytoscape.js.
 * 
 * Usage Example:
 * 
 * <topology-graph #graph></topology-graph>
 * 
 * // In your parent component:
 * @ViewChild('graph') graph!: TopologyGraphComponent;
 * 
 * // Add a process (auto-creates cloud/server if needed)
 * const processId = this.graph.add('nginx', 'server-1', 'ARM', 'aws');
 * 
 * // Add a status icon (e.g., warning)
 * this.graph.addStatusIcon(processId, this.graph.STATUS_WARNING_ICON);
 * 
 * // Remove status icon
 * this.graph.removeStatusIcon(processId);
 * 
 * // Highlight a node
 * this.graph.highlightNode(processId);
 * 
 * // Fit the view
 * this.graph.fitView();
 */

import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, inject } from '@angular/core';
import cytoscape, { Core, StylesheetCSS, NodeSingular } from 'cytoscape';
//import fcose from 'cytoscape-fcose';
import coseBilkent from 'cytoscape-cose-bilkent';
import { NgZone } from '@angular/core';

@Component({
  selector: 'sentra-node-graph',
  standalone: true,
  templateUrl: './sentra-nodes-graph.html',
  styleUrls: ['./sentra-nodes-graph.scss']
})
export class SentraNodeGraph implements AfterViewInit, OnDestroy {
  @ViewChild('cyContainer', { static: true }) cyContainer!: ElementRef<HTMLDivElement>;

  private cy!: Core;
  private  cy_layout!: cytoscape.Layouts;

  private readonly ngZone = inject(NgZone);

  private existingClouds = new Set<string>();
  private existingServers = new Set<string>();
  private processCounter = 0;

  private static CLOUD_ICON="/images/icons/cloud.svg";
  private static  SERVER_ARM_ICON="/images/icons/cloud.svg";
  private static  SERVER_INTEL_ICON="/images/icons/cloud.svg";
  private static  SERVER_AMD_ICON="/images/icons/cloud.svg";
  private static  PROCESS_ICON="/images/icons/sentra.svg";
  private static layoutOptionsClouds={
          name: 'grid',
          avoidOverlap: true,
          avoidOverlapPadding: 50,
          spacingFactor:2,
          animate: false, // Smooth transitions
          nodeDimensionsIncludeLabels: true
        } as cytoscape.LayoutOptions;

          private static layoutOptionsGrid={
          name: 'grid',
          avoidOverlap: true,
          avoidOverlapPadding: 5,
          spacingFactor:1.2,
          animate: false, // Smooth transitions
          nodeDimensionsIncludeLabels: true
        } as cytoscape.LayoutOptions;

          private static layoutOptionsProcesses={
          name: 'grid',
          avoidOverlap: true,
          avoidOverlapPadding: 5,
          spacingFactor:1,
          fit:false,
          cols:1,
          animate: false, // Smooth transitions
        } as cytoscape.LayoutOptions;

private static layoutOptions={
  name: 'cose-bilkent',
  nodeDimensionsIncludeLabels: true,
  nodeRepulsion: 70000,
  nestingFactor: 0.1,
//  idealEdgeLength: 30,
  gravityCompound: 1.0,
  gravityRangeCompound: 1.5,
  quality: 'proof',
  animate: false,
  randomize: false,
  numIter: 30000,
  tilingPaddingVertical: 70,
   tilingPaddingHorizontal: 70,
}  as cytoscape.LayoutOptions;


  ngAfterViewInit(): void {
    //this.ngZone.runOutsideAngular(() => {
    cytoscape.use(coseBilkent);
      this.cy = cytoscape({
        container: this.cyContainer.nativeElement,
        elements: [/*
          {
            data:
            {
              id: "A"
            }
          },
          {
          data:
            {
              id: "A1",
              parent: "A"
            }
          },
          {
          data:
            {
              id: "A2",
              parent: "A"
            }
          },
          {
            data:
            {
              id: "B"
            }
          },
          {
            data:
            {
              id: "B1",
              parent: "B"
            }
          },*/
        ],
        style: this.getStylesheet(),
        
      });
      
/*      this.cy_layout=this.cy.layout({
          name: 'cose',
          animate: false,
          fit: true,
          padding: 50,
          nodeRepulsion: 8000,
          idealEdgeLength: 120,
          nestingFactor: 1.2,
        });
*/
/*const layoutOptions={
          name: 'cose-bilkent',
  nodeDimensionsIncludeLabels: true, // Accounts for label sizes
  idealEdgeLength: 100, // Adjusts spacing between connected nodes
  nodeRepulsion: 4500, // Stronger repulsion to avoid overlaps
  padding: 50, // Adds extra space around the graph
  animate: false, // Smooth transitions
        } as cytoscape.LayoutOptions;*/


      // Demo: Add sample data
  /*    this.add('nginx', 'server-1', 'ARM', 'aws');
      this.add('redis', 'server-1', 'ARM', 'aws');
      this.add('postgres', 'server-2', 'Intel', 'gcp');
      this.add('api', 'server-3', 'AMD', 'azure');
      this.layout();
   */  // this.fitView();
        /*this.cy.add({
            data:
            {
              id: "A"
            }
          });
        this.cy.add(         {
          data:
            {
              id: "A1",
              parent: "A"
            }
          });
        this.cy.add(         {
          data:
            {
              id: "A1-1",
              parent: "A1"
            }
          });*/
    //  this.cy_layout.run();
//      this.addStatusIcon('server-1-nginx', this.STATUS_WARNING_ICON);
 //     this.addStatusIcon('server-2-postgres', this.STATUS_OK_ICON);
    
    //);
    
  }
  layout() {
/*    this.cy.nodes("[type=='process'").layout(SentraNodeGraph.layoutOptionsProcesses).run();
    this.cy.nodes("[type=='server'").layout(SentraNodeGraph.layoutOptions).run();
    this.cy.nodes("[type=='cloud'").layout(SentraNodeGraph.layoutOptionsClouds).run();*/
    this.cy.layout(SentraNodeGraph.layoutOptions).run();
//    this.cy.layout(SentraNodeGraph.layoutOptionsGrid).run();
  }

  ngOnDestroy(): void {
    if (this.cy) {
      this.cy.destroy();
    }
  }

  /**
   * Add a process to the topology.
   * @param processName Name of the process (e.g., 'nginx')
   * @param serverId ID of the server (e.g., 'server-1')
   * @param architecture 'ARM' | 'Intel' | 'AMD'
   * @param cloudId ID of the cloud (e.g., 'aws')
   * @returns The generated process node ID
   */
  public add(
    processId: string,
    processName: string,
    serverId: string,
    architecture: string, //'ARM' | 'Intel' | 'AMD',
    cloudId: string
  ): string {
    if (!this.cy) return '';

    // 1. Add cloud node if missing
    if (!this.existingClouds.has(cloudId)) {
      this.cy.add({
        group: 'nodes',
        data: {
          id: cloudId,
          label: cloudId,
          type: 'cloud'
        }
      });
      this.existingClouds.add(cloudId);
    }

    // 2. Add server node if missing
    let si=cloudId+"-"+serverId
    if (!this.existingServers.has(si)) {
      this.cy.add({
        group: 'nodes',
        data: {
          id: si,
          label: serverId,
          type: 'server',
          parent: cloudId,
          architecture
        }
      });
      this.existingServers.add(si);
    }

    // 3. Add process node (unique id: serverId-processName)
    if (this.cy.getElementById(processId).empty()) {
      this.cy.add({
        group: 'nodes',
        data: {
          id: processId,
          label: processName,
          type: 'process',
          parent: si
        }
      });
    }

    // 4. Rerun layout for updated positioning
   /* this.cy.layout({
      name: 'cose',
      animate: true,
      fit: true,
      padding: 50,
      nodeRepulsion: 8000,
      idealEdgeLength: 120,
      nestingFactor: 1.2,
    }).run();*/

    return processId;
  }

  

  /**
   * Highlight a node by ID (adds a glow effect).
   * @param nodeId Node ID to highlight
   */
  public setAttested(nodeId: string,bAttested:boolean): void {
    if (!this.cy) return;
    const node = this.cy.getElementById(nodeId);
    if (node) {
        if(bAttested)
        { 
          node.removeClass('attestation-failed');
          node.addClass('attested');
        }
        else
        {
          node.removeClass('attested');
          node.addClass('attestation-failed');

        }
    }
  }

  public setCommitteeMember(nodeId: string,bMember:boolean): void {
    if (!this.cy) return;
    const node = this.cy.getElementById(nodeId);
    if (node) {
        if(bMember)
        { 
          node.addClass('committee-member');
        }
        else
        {
          node.removeClass('committee-member');
        }
    }
  }

  /**
   * Fit the Cytoscape view to all elements.
   */
  public fitView(): void {
    if (this.cy) {
      this.cy.fit(undefined, 50);
    }
  }

  /**
   * Cytoscape stylesheet for all node types and overlays.
   */
  private getStylesheet(): StylesheetCSS[] {
    return [
      // Cloud nodes (compound)
      {
        selector: 'node[type="cloud"]',
        css: {
          'shape': 'roundrectangle',
          'background-color': '#e3f2fd',
          'border-width': 4,
          'border-color': '#1976d2',
/*          'background-image': SentraNodeGraph.CLOUD_ICON,
          'background-fit': 'contain',
          'background-position-x': '50%',
          'background-position-y': '50%',
          'background-width': '100%',
          'background-height': '100%',
 */         'label': 'data(label)',
          'font-size': 20,
          'font-weight': 'bold',
          'text-valign': 'bottom',
          'text-halign': 'center',
          'color': '#1976d2',
          'padding': '40px',
          'z-index': 1,
                    'compound-sizing-wrt-labels':'include',
                    'text-margin-y':-20,

        }
      },
      // Server nodes (compound)
      {
        selector: 'node[type="server"][architecture="ARM"]',
        css: {
          'shape': 'roundrectangle',
          'background-color': '#e8f5e9',
          'border-width': 3,
          'border-color': '#388e3c',
          'label': 'data(label)',
          'font-size': 16,
          'font-weight': 'bold',
          'text-valign': 'bottom',
          'text-halign': 'center',
          'color': '#388e3c',
          'padding': '24px',
                    'text-margin-y':5,
          'z-index': 2,
                    'compound-sizing-wrt-labels':'include'

        }
      },
      {
        selector: 'node[type="server"][architecture="Intel"]',
        css: {
          'shape': 'roundrectangle',
          'background-color': '#e3f2fd',
          'border-width': 3,
          'border-color': '#1976d2',
          'label': 'data(label)',
          'font-size': 16,
          'font-weight': 'bold',
          'text-valign': 'bottom',
          'text-halign': 'center',
          'color': '#1976d2',
          'padding': '24px',
          'z-index': 2,
                    'text-margin-y':5,
          'compound-sizing-wrt-labels':'include'
        }
      },
      {
        selector: 'node[type="server"][architecture="AMD"]',
        css: {
          'shape': 'roundrectangle',
          'background-color': '#fff3e0',
          'border-width': 3,
          'border-color': '#f57c00',
          'label': 'data(label)',
          'font-size': 16,
          'font-weight': 'bold',
          'text-valign': 'bottom',
          'text-halign': 'center',
          'color': '#f57c00',
          'padding': '24px',
          'z-index': 2,
                    'text-margin-y':5,
          'compound-sizing-wrt-labels':'include'
        }
      },
      // Process nodes (leaf)
      {
        selector: 'node[type="process"]',
        css: {
          'shape': 'ellipse',
          'width': 48,
          'height': 48,
          'background-color': '#fffde7',
          'border-width': 2,
          'border-color': '#fbc02d',
          'background-image': SentraNodeGraph.PROCESS_ICON,
          'background-fit': 'cover',
          'background-opacity': 1,
          'label': 'data(label)',
          'font-size': 13,
          'font-weight': 'bold',
          'text-valign': 'bottom',
          'text-halign': 'center',
          'color': '#fbc02d',
          'text-margin-y': 8,
          // Status badge overlay (background-image-2)
//          'background-image-2': '',
 //         'background-fit-2': 'contain',
  //        'background-width-2': '40%',
    //      'background-height-2': '40%',
      //    'background-position-x-2': '80%',
        //  'background-position-y-2': '80%',
          //'background-opacity-2': 1,
          'z-index': 3
        }
      },
      // Highlighted node
      {
        selector: '.attested',
        css: {
//          'box-shadow': '0 0 16px 8px #1976d2',
          'border-color': '#0ee232',
          'border-width': 6
        }
      },
      {
        selector: '.attestation-failed',
        css: {
//          'box-shadow': '0 0 16px 8px #1976d2',
          'border-color': '#e11013',
          'border-width': 6
        }

      },
            {
        selector: '.committee-member',
        css: {
          'text-background-color':'#4686d96e',
          'text-background-opacity':1.0,
          'text-background-shape':'rectangle',
          'text-background-padding':'2px'
        }
      },

    ];
  }
}
