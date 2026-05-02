import { ChangeDetectorRef, Component, ViewChild } from '@angular/core';
import { ImageSelector } from "../image-selector/image-selector";
import {MatGridListModule} from '@angular/material/grid-list';
import { MatButtonModule } from '@angular/material/button';
import { ClassificationResult, InferenceResult, RestService } from '../../rest.service';
import { PredictionResultDisplay } from '../prediction-result-display/prediction-result-display';
import { ImageSelectedEvent, MnistNumberSelector } from "../mnistnumber-selector/mnistnumber-selector";
import { AiWidgetComponent } from "../ai-widget.component/ai-widget.component";
import { MatCardModule } from "@angular/material/card";
import {MatDividerModule} from '@angular/material/divider';
import { CommonModule } from '@angular/common';
@Component({
  selector: 'app-sentra-client-dashboard',
  imports: [CommonModule,MnistNumberSelector, MatDividerModule, MatGridListModule, MatButtonModule,
     PredictionResultDisplay, MnistNumberSelector, AiWidgetComponent, MatCardModule],
  templateUrl: './sentra-client-dashboard.html',
  styleUrl: './sentra-client-dashboard.css',
})
export class SentraClientDashboard 
{
predictionProbabilities: number[]=[];
predictionResult: number=0;
selectedImage:ImageSelectedEvent|null=null;

 posRight_MovingDigitImage:number=180;
isAIDisabled: boolean=true;


constructor(private m_RestService: RestService,private cdr: ChangeDetectorRef)
  {
  }
  onDoResetSentra()
  {
    this.m_RestService.doReset().subscribe();

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
doInference() 
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
   this.showInferenceResult(inferenceResult);
  });
}
else if(this.selectedImage.image_data!=null)
{
  this.m_RestService.doPredictionPixel(this.selectedImage.image_data).subscribe((inferenceResult)=>
  {
   this.showInferenceResult(inferenceResult);
  });
}
}

doDistributeShares()
{
  console.log("Distribute shares");
}

doRemoteAttestation()
  {
    this.m_RestService.doRemoteAttestation().subscribe();
  }

  doCommitteeSelection()
  {
    this.m_RestService.doCommitteeSelection().subscribe();
  }

  doNodeSelection()
  {
    this.m_RestService.doNodeSelection("1").subscribe();
  }

  doImageUpload()
  {
  const divImg = document.getElementById('moving-digit-image');
  if(divImg===null)
    return;

  // Show the circle
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
    divImg.style.display='none';
    if(this.selectedImage?.image_b64!=null)
    this.m_RestService.doImageUpload(this.selectedImage?.image_b64).subscribe();
  };
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
