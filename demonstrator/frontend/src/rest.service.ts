import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export class SentraNode
{
  public node_id: string="";
  public host: string="";
  public cpu: string="";
  public operator: string="";
  public attested:boolean=false;
}


@Injectable({
  providedIn: 'root'
})
export class RestService {

  constructor(private m_HttpClient:HttpClient) { }



  getNodes():Observable<SentraNode[]>
   {
    return this.m_HttpClient.get<SentraNode[]>('/api/v1/getNodes');
   }
  


}
