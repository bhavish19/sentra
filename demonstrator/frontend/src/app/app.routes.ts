import { Routes } from '@angular/router';
import { SentraNodesTable } from './sentra-nodes-table/sentra-nodes-table';
import { SentraCloudDashboard } from './sentra-cloud-dashboard/sentra-cloud-dashboard';
import { SentraClientDashboard } from './sentra-client-dashboard/sentra-client-dashboard';

export const routes: Routes = [

  {path: 'sentra-nodes-table', component: SentraNodesTable,title:'SENTRA Backend'},
  {path: 'sentra-nodes', component: SentraCloudDashboard,title:'SENTRA Backend'},
      {path: '**', component: SentraClientDashboard,title:'SENTRA Frontend'}  
];
