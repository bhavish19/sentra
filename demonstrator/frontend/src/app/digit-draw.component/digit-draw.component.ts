import { CommonModule } from '@angular/common';
import {
  Component,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RestService } from '../../rest.service';

@Component({
  imports:[CommonModule,FormsModule],
  selector: 'app-digit-draw',
  templateUrl: './digit-draw.component.html',
  styleUrls: ['./digit-draw.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DigitDrawComponent {

 transposeMatrix = (matrix: number[][]): number[][] => {
  const rows = matrix.length;
  const cols = matrix[0].length;

  // Initialize the result matrix with empty arrays
  const result: number[][] = Array.from({ length: cols }, () => []);

  // Populate the result matrix by swapping rows and columns
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }

  return result;
};

 
  grid: number[][] = Array.from({ length: 28 }, () => Array(28).fill(0)); // 28x28 grid
  isDrawing = false; // Tracks whether the user is drawing
  brushSize = 2; // Brush radius (in cells)
  maxIntensity = 255; // Maximum intensity for the center cell

  constructor(private cdr: ChangeDetectorRef,private m_RestService: RestService) {}

  // Start drawing
  onMouseDown(row: number, col: number): void {
    this.isDrawing = true;
    this.paintCell(row, col);
  }

  // Stop drawing
  onMouseUp(): void {
    this.isDrawing = false;
  }

  // Draw while moving the mouse
  onMouseMove(row: number, col: number): void {
    if (this.isDrawing) {
      this.paintCell(row, col);
    }
  }

  // Paint a cell and its neighbors with a brush effect
  private paintCell(centerX:number, centerY:number): void {
  const gridWidth = 28; // Assuming a 28x28 grid
  const gridHeight = 28;


  // Iterate over the cells within the brush radius
  for (let y = -this.brushSize; y <= this.brushSize; y++) {
    for (let x = -this.brushSize; x <= this.brushSize; x++) {
      const targetX = centerX + x;
      const targetY = centerY + y;

      // Skip out-of-bounds cells
      if (targetX < 0 || targetX >= gridWidth || targetY < 0 || targetY >= gridHeight) {
        continue;
      }

      // Calculate the distance from the brush center
      const distance = Math.sqrt(x * x + y * y);

      // Skip cells outside the brush radius
      if (distance > this.brushSize) {
        continue;
      }

      // Calculate the brush intensity (e.g., linear falloff)
      const intensity = Math.max(0, 0.5 - distance / this.brushSize);

      // Calculate the target cell index
      //const targetIndex = targetY * gridWidth + targetX;

      // Additive blending: Add the intensity to the existing value
      const updatedValue = this.grid[targetX][targetY] + intensity * 255;

      // Clamp the value to the maximum (255 for grayscale)
      this.grid[targetX][targetY]  = Math.min(255, updatedValue);
    }
  }

  // Trigger change detection
  this.cdr.markForCheck();
}


  // Clear the grid
  clearGrid(): void {
    this.grid = Array.from({ length: 28 }, () => Array(28).fill(0));
    this.cdr.markForCheck();
  }

  // Export the grid as a normalized Float32Array
  getImageData(): Float32Array {
    return new Float32Array(this.transposeMatrix(this.grid).flat().map((v) => v / 255));
  }
}
