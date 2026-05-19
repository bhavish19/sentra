import { AfterViewInit, ChangeDetectorRef, Component, ElementRef, HostListener, ViewChild } from '@angular/core';
import { ImageSelector } from "../image-selector/image-selector";
import {MatGridListModule} from '@angular/material/grid-list';
import { MatButtonModule } from '@angular/material/button';
import { ClassificationResult, InferenceResult, RestService, SentraNode } from '../../rest.service';
import { PredictionResultDisplay } from '../prediction-result-display/prediction-result-display';
import { ImageSelectedEvent, MnistNumberSelector } from "../mnistnumber-selector/mnistnumber-selector";
import { AiWidgetComponent } from "../ai-widget.component/ai-widget.component";
import { MatCardModule } from "@angular/material/card";
import {MatDividerModule} from '@angular/material/divider';
import { CommonModule } from '@angular/common';
import { createWebSocketURLForPath, float32ArrayToImageUrl, generateBase64Image, generateFlashingArrowSVG, generateRandomFloat32Array, IArrowAnimation, sleep } from '../../utils';
@Component({
  selector: 'app-sentra-client-dashboard',
  imports: [CommonModule,MnistNumberSelector, MatDividerModule, MatGridListModule, MatButtonModule,
     PredictionResultDisplay, MnistNumberSelector, AiWidgetComponent, MatCardModule],
  templateUrl: './sentra-client-dashboard.html',
  styleUrl: './sentra-client-dashboard.scss',
})
export class SentraClientDashboard implements AfterViewInit
{
 @ViewChild('arrowRight', { static: true }) divArrowRight! :ElementRef<HTMLDivElement>;
 @ViewChild('arrowLeft', { static: true }) divArrowLeft! :ElementRef<HTMLDivElement>;
 @ViewChild('cardDigit',{ static: true }) htmlCardDigit! :ElementRef;
 @ViewChild('cardInference',{ static: true }) htmlCardInference! :ElementRef;

   m_socket?:WebSocket=undefined;
predictionProbabilities: number[]=[];
predictionResult: number=0;
selectedImage:ImageSelectedEvent|null=null;

 posRight_MovingDigitImage:number=180;
isAIDisabled: boolean=true;
m_Committee:SentraNode[]|undefined=undefined;
m_InferenceResult:InferenceResult|null=null;

constructor(private m_RestService: RestService,private cdr: ChangeDetectorRef)
  {
    this.receiveWebSocketMsg();
    this.autoWebSocketReconnect();

  }


 @HostListener('window:resize', ['$event'])
  onResize(event: Event) {
    this.ngAfterViewInit();
  }

  ngAfterViewInit(): void {
    let divAi=document.getElementById("ai_widget") as HTMLElement;
    let rectAi=divAi.getBoundingClientRect();
    let rectDigit=this.htmlCardDigit.nativeElement.getBoundingClientRect();
    let rectInference=this.htmlCardInference.nativeElement.getBoundingClientRect();
    let dx:number=rectAi.width/2+rectAi.left-rectDigit.right;
    let dy:number=rectAi.top-(rectDigit.top+20);
    let arrow:IArrowAnimation=generateFlashingArrowSVG(dx,dy,false,0);
    this.divArrowRight.nativeElement.innerHTML=arrow.svg;
    this.divArrowRight.nativeElement.style.left=rectDigit.right+20+"px";
    this.divArrowRight.nativeElement.style.top=rectDigit.top+20+"px";

    arrow=generateFlashingArrowSVG(dx,dy,true,arrow.duration+1);
    this.divArrowLeft.nativeElement.innerHTML=arrow.svg;
    this.divArrowLeft.nativeElement.style.left=rectInference.right+20+"px";
    this.divArrowLeft.nativeElement.style.top=rectAi.bottom+10+"px";
  }
  onDoResetSentra()
  {
    this.m_RestService.doReset().subscribe();

  }

doRestart()
{
  this.onDoResetSentra();
  window.location.reload();
}

onImageSelected(event: ImageSelectedEvent) 
{
  this.selectedImage=event;
  this.isAIDisabled=false;
  this.cdr.detectChanges();
}


 getClassificationLikelihoods(classificationResults:ClassificationResult[]): number[] {
    let ret:number[]=[];
    for (const c of classificationResults)
    {
      ret.push(c.likelihood);
    }
    return ret;
  }  

getFileNameWithoutExtension(path: string): string {
  // Find the last index of either '\' or '/'
  const lastBackslashIndex = path.lastIndexOf('\\');
  const lastSlashIndex = path.lastIndexOf('/');
  const lastIndex = Math.max(lastBackslashIndex, lastSlashIndex);

  // Extract the file name
  const fileNameWithExtension = path.substring(lastIndex + 1);

  // Remove the extension
  const lastDotIndex = fileNameWithExtension.lastIndexOf('.');
  return lastDotIndex !== -1
    ? fileNameWithExtension.substring(0, lastDotIndex)
    : fileNameWithExtension;
}

showInferenceResult(inferenceResult:InferenceResult)
{
    console.log(inferenceResult);
    this.predictionResult=inferenceResult.predictedValue;
    this.predictionProbabilities=this.getClassificationLikelihoods(inferenceResult.classificationResults);
    this.cdr.detectChanges();
}
doInference(bPlain:boolean) 
{
/*  let fileName:string|null=this.m_imageSelector.imageName;
  if(fileName===null)
    return;
  fileName=this.getFileNameWithoutExtension(fileName);
  let index:number=Number(fileName);
  */
 if(this.selectedImage==null)
  return;
if(this.selectedImage.index!=null)
{
  this.m_RestService.doPrediction(this.selectedImage.index).subscribe((inferenceResult)=>
  {
    this.m_InferenceResult=inferenceResult;
    console.log("Inference result is now: ",this.m_InferenceResult);
    if(bPlain)
    {
      let imgResult:string|null=generateBase64Image(inferenceResult.predictedValue);
      if(imgResult!=null)
        this.m_RestService.doReceiveResult(imgResult,"").subscribe();
      else
        this.showInferenceResult(inferenceResult);
    }
    else
        this.showInferenceResult(inferenceResult);
  });
}
else if(this.selectedImage.image_data!=null)
{
  console.log("Do Inference for Handwritten Image");
  this.m_RestService.doPredictionPixel(this.selectedImage.image_data).subscribe((inferenceResult)=>
  {
    this.m_InferenceResult=inferenceResult;
    console.log("Inference result is now: ",this.m_InferenceResult);
    if(bPlain)
    {
      let imgResult:string|null=generateBase64Image(inferenceResult.predictedValue);
      if(imgResult!=null)
        this.m_RestService.doReceiveResult(imgResult,"").subscribe();
      else
        this.showInferenceResult(inferenceResult);
    }else{
   this.showInferenceResult(inferenceResult);}
  });
}
}

async doReceiveShares()
{
  console.log("Receive shares");
  if(this.m_Committee===undefined)
    return;
  let numShares:number=this.m_Committee.length;
  for(let i=0;i<numShares;i++)
    {
      const randImgData:Float32Array=generateRandomFloat32Array(784);
      const img:string=float32ArrayToImageUrl(randImgData,28,28,i);
      this.m_RestService.doReceiveResult(img,this.m_Committee[i].node_id).subscribe();
      await sleep(1000);
    }

}

async doExecuteMPC()
{
  this.m_RestService.doExecuteMPC().subscribe();
  await sleep(8000);
  this.m_RestService.doStopMPC().subscribe();
  await this.doReceiveShares();
  await sleep(4000);
  this.doInference(false);
}

async doDistributeShares()
{
  console.log("Distribute shares");
  if(this.m_Committee===undefined)
    return;
  let numShares:number=this.m_Committee.length;
  for(let i=0;i<numShares;i++)
    {
      const randImgData:Float32Array=generateRandomFloat32Array(784);
      const img:string=float32ArrayToImageUrl(randImgData,28,28,i);
      this.animateImageUpload(img,this.m_Committee[i].node_id);
      await sleep(1000);
    }
}

doRemoteAttestation()
  {
    this.m_RestService.doRemoteAttestation().subscribe();
  }

  doCommitteeSelection()
  {
    this.m_RestService.doCommitteeSelection().subscribe(
      (res)=>
      {
        console.log("Recevied Committee: ",res);
        this.m_Committee=res;
      }
    );
  }

  doNodeSelection()
  {
    this.m_RestService.doNodeSelection("-").subscribe();
  }

setPosRight_MovingDigitImage()
{
  let div=document.getElementById("ai_widget") as HTMLElement;
  let rect=div.getBoundingClientRect();
  console.log("Rect: ",rect);
  this.posRight_MovingDigitImage= window.innerWidth-rect.right;
}

animateImageUpload(image:string|null,node_id:string|null)
{
  if(image===null)
    return;
  const divImgOrig = document.getElementById('moving-digit-image-div');
  if(divImgOrig===null)
    return;
 const hmtlImgOrig:HTMLImageElement|null = document.getElementById('moving-digit-image-image') as HTMLImageElement;
  if(hmtlImgOrig===null)
    return;
  const divImg=divImgOrig.cloneNode(false) as HTMLElement;
  
  divImgOrig.insertAdjacentElement('afterend', divImg);
  const hmtlImg=hmtlImgOrig.cloneNode(false) as HTMLIFrameElement
  divImg.appendChild(hmtlImg);

  this.setPosRight_MovingDigitImage();
  divImg.style.right=this.posRight_MovingDigitImage+"px";

  hmtlImg.src=image;
  // Show 
  divImg.style.display = 'block';


  const animation:Animation = divImg.animate(
      [
        { transform: 'translate(0px,0px)' },
        { transform: 'translate('+(this.posRight_MovingDigitImage+56)+'px,0px)' },

      ],
      {
        duration: 2000,
        easing: 'linear',
        iterations:1,
      }
    );
  animation.onfinish=(e)=>
  {
    if(divImg.parentNode)
      divImg.parentNode.removeChild(divImg);
    if(image!=null)
      this.m_RestService.doImageUpload(image,node_id).subscribe();
  };
}

animateReceiveResult(image:string)
{
  const divImgOrig = document.getElementById('moving-result-div');
  if(divImgOrig===null)
    return;
 const hmtlImgOrig:HTMLImageElement|null = document.getElementById('moving-result-image') as HTMLImageElement;
  if(hmtlImgOrig===null)
    return;
  const divImg=divImgOrig.cloneNode(false) as HTMLElement;
  
  divImgOrig.insertAdjacentElement('afterend', divImg);
  const hmtlImg=hmtlImgOrig.cloneNode(false) as HTMLIFrameElement
  divImg.appendChild(hmtlImg);

  this.setPosRight_MovingDigitImage();
  divImg.style.right=this.posRight_MovingDigitImage+"px";

  hmtlImg.src=image;
  // Show 
  divImg.style.display = 'block';


  const animation:Animation = divImg.animate(
      [
        { transform: 'translate('+(this.posRight_MovingDigitImage+56)+'px,0px)' },
        { transform: 'translate(0px,0px)' },
      ],
      {
        duration: 2000,
        easing: 'linear',
        iterations:1,
      }
    );
  animation.onfinish=(e)=>
  {
    if(divImg.parentNode)
      divImg.parentNode.removeChild(divImg);
  };
}

  doImageUpload()
  {
    if(this.selectedImage)
  this.animateImageUpload(this.selectedImage.image_b64,null);
}

async handleReceiveResult(resultMsg:any)
{
    console.log("doReceiveResult() - recevied message: ",resultMsg);
    let img=resultMsg.receiveResultOnClient;
    let node_id=resultMsg.node_id;
    console.log("doReceiveResult() - received message for node: ",node_id);
    this.animateReceiveResult(img);
    console.log("Inference Result is still: ",this.m_InferenceResult)
    if(node_id==="" && this.m_InferenceResult!=null)
    {
      await sleep(2000);
      this.showInferenceResult(this.m_InferenceResult);
    }

}

  //do something with the web socket
receiveWebSocketMsg()
{
  if(this.m_socket&&(this.m_socket.readyState === WebSocket.OPEN || this.m_socket.readyState === WebSocket.CONNECTING))
    return; //do nothing if already connected
  this.m_socket= new WebSocket(createWebSocketURLForPath("/wsclient"));
  this.m_socket.onerror=((ev:any)=>
    {
    }
  );
  this.m_socket.onclose=((ev:any)=>
    {
    }
  );

  this.m_socket.onmessage=(async (ev:any)=>
  {
    const obj = JSON.parse(ev.data);
    if('receiveResultOnClient' in obj) //Cloud has changed
    {
      await this.handleReceiveResult(obj);
    }
    console.log(ev.data)
  });
}

autoWebSocketReconnect()
{
  setInterval(()=>this.receiveWebSocketMsg(),10000)
}



}

/*
 doInference() {
    let z=this.transposeMatrix(this.grid);
    let g=new Float32Array(z.flat().map((v) => v / 255));
  this.m_RestService.doPredictionPixel(g).subscribe
  (
    (result)=>
    {
      console.log(result);
    }
  )
}*/
