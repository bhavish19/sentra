import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { MatTableDataSource, MatTableModule } from '@angular/material/table';
import { RestService, SentraNode } from '../rest.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet,MatTableModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {

  // Define which columns to display
  displayedColumns: string[] = ['node_id','host','cpu', 'operator','attested'];
  dataSource = new MatTableDataSource<SentraNode>([]);
  protected readonly title = signal('frontend');

  constructor(private m_RestService: RestService)
  {
    this.m_RestService.getNodes().subscribe((nodes)=>{this.dataSource.data=nodes;
      console.log("Loaded nodes: "+nodes.length);
    });
  }
}
