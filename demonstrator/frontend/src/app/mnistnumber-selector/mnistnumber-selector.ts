import { ChangeDetectorRef, Component, EventEmitter, OnInit, Output } from '@angular/core';
import { MnistImage, RestService } from '../../rest.service';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatSelectModule } from '@angular/material/select';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';

export interface ImageSelectedEvent {
  digit: number;
  index: number;
}

@Component({
  imports: [MatButtonModule,MatProgressSpinnerModule,MatFormFieldModule,MatGridListModule,MatSelectModule,FormsModule,MatCardModule,CommonModule],
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
  loading = false;

  @Output() imageSelected = new EventEmitter<number>();

  constructor(private m_RestService: RestService,private cdr: ChangeDetectorRef) {}

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
    this.imageSelected.emit(image.index);
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
}
