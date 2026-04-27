import { Component, signal } from '@angular/core';
import { RestService, SentraNode } from '../../rest.service';
import { MatTableDataSource } from '@angular/material/table';
import { MatTableModule } from '@angular/material/table';
@Component({
  selector: 'app-sentra-nodes-table',
  imports: [MatTableModule],
  templateUrl: './sentra-nodes-table.html',
  styleUrl: './sentra-nodes-table.css',
})
export class SentraNodesTable {

  // Define which columns to display
  displayedColumns: string[] = ['node_id','host','grpc_url','cpu', 'operator','attested'];
  dataSource = new MatTableDataSource<SentraNode>([]);
  protected readonly title = signal('frontend');

  constructor(private m_RestService: RestService)
  {
    this.m_RestService.getNodes().subscribe((nodes)=>{this.dataSource.data=nodes;
      console.log("Loaded nodes: "+nodes.length);
    });
  }
}
