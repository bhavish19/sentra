import { ChangeDetectorRef, Component, ViewChild } from '@angular/core';
import { ImageSelector } from "../image-selector/image-selector";
import {MatGridListModule} from '@angular/material/grid-list';
import { MatButtonModule } from '@angular/material/button';
import { ClassificationResult, RestService } from '../../rest.service';
import { PredictionResultDisplay } from '../prediction-result-display/prediction-result-display';
import { MnistNumberSelector } from "../mnistnumber-selector/mnistnumber-selector";
import { AiWidgetComponent } from "../ai-widget.component/ai-widget.component";

@Component({
  selector: 'app-sentra-client-dashboard',
  imports: [MnistNumberSelector, MatGridListModule, MatButtonModule, PredictionResultDisplay, MnistNumberSelector, AiWidgetComponent],
  templateUrl: './sentra-client-dashboard.html',
  styleUrl: './sentra-client-dashboard.css',
})
export class SentraClientDashboard 
{
predictionProbabilities: number[]=[];
predictionResult: number=0;
selectedImageIndex:number=-1;

constructor(private m_RestService: RestService,private cdr: ChangeDetectorRef)
  {
  }
  onDoResetSentra()
  {
    this.m_RestService.doReset().subscribe();

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

onDoInference() 
{
/*  let fileName:string|null=this.m_imageSelector.imageName;
  if(fileName===null)
    return;
  fileName=this.getFileNameWithoutExtension(fileName);
  let index:number=Number(fileName);
  */
 if(this.selectedImageIndex==-1)
  return;
  this.m_RestService.doPrediction(this.selectedImageIndex).subscribe((inferenceResult)=>
  {
    console.log(inferenceResult);
    this.predictionResult=inferenceResult.predictedValue;
    this.predictionProbabilities=this.getClassificationLikelihoods(inferenceResult.classificationResults);
    this.cdr.detectChanges();
  });
}

}
