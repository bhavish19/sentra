import { ChangeDetectorRef, Component, EventEmitter, inject, OnInit, Output, ViewChild } from '@angular/core';
import { MnistImage, RestService } from '../../rest.service';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatSelectModule } from '@angular/material/select';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { DigitDrawComponent } from "../digit-draw.component/digit-draw.component";
import { MatDialog, MatDialogRef, MatDialogContent, MatDialogActions }  from '@angular/material/dialog';
import { float32ArrayToImageUrl } from '../../utils';
export class ImageSelectedEvent {
  index: number|null=null;
  image_data:Float32Array|null=null;
  image_b64:string|null=null;
}

@Component({
  imports: [MatButtonModule, MatProgressSpinnerModule, MatFormFieldModule, MatGridListModule, 
    MatSelectModule, FormsModule, MatCardModule, CommonModule],
  selector: 'mnist-number-selector',
  templateUrl: './mnistnumber-selector.html',
  styleUrl: './mnistnumber-selector.css'
})

export class MnistNumberSelector {

  digits = Array.from({ length: 10 }, (_, i) => i);
  view: 'digit' | 'images' | 'selection_start' | 'digit_image' = 'selection_start';
  images: MnistImage[] = [];
  selectedDigit: number | null = null;
  selectedIndex: number | null = null;
  selectedImage:string |null=null;
  selectedImageData:Float32Array|null=null;
  loading = false;

  @Output() imageSelected = new EventEmitter<ImageSelectedEvent>();
  readonly m_dlgDrawDigit = inject(MatDialog);
  constructor(private m_RestService: RestService,private cdr: ChangeDetectorRef) 
  {
  }

  selectDigit(digit: number) {
    this.selectedDigit = digit;
    this.loading = true;
    this.m_RestService.getTestImagesForDigit(digit).subscribe(images => {
      this.images = images;
      this.view = 'images';
      this.loading = false;
      this.selectedIndex = null;
      console.log("Image loading done");
      this.cdr.detectChanges();

    });
  }

  selectImage(image: MnistImage) {
    this.selectedIndex = image.index;
    this.selectedImage=image.image_b64;
    this.view='digit_image';
    let event=new ImageSelectedEvent();
    event.index=image.index
    event.image_b64=this.selectedImage;
    this.imageSelected.emit(event);
  }

  backToDigits() {
    this.view = 'digit';
    this.images = [];
    this.selectedDigit = null;
    this.selectedIndex = null;
  }

  onStartDigitSelection() 
    {
      this.view='digit';
    }

    onDrawDigit() 
    {
      console.log("onDrawDigit()");
      let dialogRef=this.m_dlgDrawDigit.open(DigitDrawDialog);
      dialogRef.afterClosed().subscribe(result => {
      console.log('The dialog was closed');
      if (result !== undefined) {
        console.log(result);
        this.selectedImageData=result;
        this.selectedIndex=null;
        this.selectedImage=float32ArrayToImageUrl(result,28,28,0);
        this.view='digit_image';
        this.cdr.detectChanges();
        let event=new ImageSelectedEvent();
        event.image_data=result;
        event.image_b64=this.selectedImage;
        this.imageSelected.emit(event);
      }
    });
}

}

@Component({
  selector: 'digit-draw-dialog',
  templateUrl: 'digit-draw-dialog.html',
  imports: [
    DigitDrawComponent,
    MatDialogContent,
    MatDialogActions,
    MatButtonModule
],
})
export class DigitDrawDialog {
  @ViewChild('digit_draw_component') m_DigitDrawComponent!: DigitDrawComponent;

onOkClick() {
    this.dialogRef.close(this.m_DigitDrawComponent.getImageData());
}
onCancelClick() {
    this.dialogRef.close();
}
  readonly dialogRef = inject(MatDialogRef<DigitDrawDialog>);

}
