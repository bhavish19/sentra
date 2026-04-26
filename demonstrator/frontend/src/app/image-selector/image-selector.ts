import { Component,ChangeDetectorRef } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';

@Component({
  selector: 'app-image-selector',
    imports: [MatButtonModule,MatCardModule],
  templateUrl: './image-selector.html',
  styleUrls: ['./image-selector.css']
})
export class ImageSelector {
  imageSrc: string | ArrayBuffer | null = null;

  imageName:string|null=null;

   constructor(private cdr: ChangeDetectorRef) {}

  // Trigger the hidden file input
  triggerFileInput(): void {
    const fileInput = document.getElementById('fileInput') as HTMLInputElement;
    fileInput.click();
  }

  // Handle file selection and preview
  onFileSelected(event: Event): void {
    console.log("On file selected")
    const input = event.target as HTMLInputElement;

    if (input.files && input.files[0]) {
      const file = input.files[0];
      this.imageName=file.name;
      const reader = new FileReader();

      reader.onload = () => {
        this.imageSrc = reader.result; // Set the image source for preview
        console.log("Image data read");
        this.cdr.detectChanges();
      };

      reader.readAsDataURL(file); // Read the file as a data URL
    }
  }
}
