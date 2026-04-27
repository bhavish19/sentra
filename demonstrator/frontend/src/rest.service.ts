import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export class SentraNode
{
  public node_id: string="";
  public label:string=""; 
  public host: string="";
  public grpc_url: string="";
  public cpu: string="";
  public operator: string="";
  public attested:boolean=false;
}

export class ClassificationResult
{
  public predictedValue:number=-1;
  public likelihood:number=0.0;
}
export class InferenceResult
{
  getClassificationLikelihoods(): number[] {
    let ret:number[]=[];
    for (const c of this.classificationResults)
    {
      ret.push(c.likelihood);
    }
    return ret;
    
  }
  public predictedValue:number=-1;
  public classificationResults:ClassificationResult[]=[];
}


export interface MnistImage {
  index: number;
  image_b64: string;
}

export interface MnistImageDetail {
  index: number;
  label: number;
  image_b64: string;
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
  
  getCommittee() :Observable<SentraNode[]>
   {
    return this.m_HttpClient.get<SentraNode[]>('/api/v1/getCommittee');
   }

   doReset():Observable<Object>
   {
    return this.m_HttpClient.post('api/v1/postReset',null);
   }

   doPrediction(index:number):Observable<InferenceResult>
   {
    return this.m_HttpClient.get<InferenceResult>('/api/v1/getPrediction/'+index);

   }

   /** Returns all test-set images for the given digit. */
  getTestImagesForDigit(digit: number): Observable<MnistImage[]> {
    return this.m_HttpClient.get<MnistImage[]>('/api/v1/mnist/getTestImagesForDigit/'+digit);
  }

  /** Returns a single image by its index in the MNIST test set. */
  getTestImageForIndex(index: number): Observable<MnistImageDetail> {
    return this.m_HttpClient.get<MnistImageDetail>('/api/v1/mnist/getTestImageForIndex/'+index);
  }


}
