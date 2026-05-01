import {
  Component,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
} from '@angular/core';

import { CommonModule } from '@angular/common';
import { RestService } from '../../rest.service';

@Component({
  imports:[CommonModule],
  selector: 'app-digit-draw',
  templateUrl: './digit-draw.component.html',
  styleUrls: ['./digit-draw.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DigitDrawComponent {
  /** Flat 784-element array; each value is 0 (white) or 255 (black). */
  grid: number[] = new Array(784).fill(0);

  /** Convenience array of indices [0..783] used by *ngFor in the template. */
  readonly indices: number[] = Array.from({ length: 784 }, (_, i) => i);

  /** True while the primary mouse button is held down over the canvas. */
  isDrawing = false;

  constructor(private cdr: ChangeDetectorRef,private m_RestService: RestService) {}

  // ── Drawing events ────────────────────────────────────────────────────────

  onMouseDown(index: number): void {
    this.isDrawing = true;
    this.paintCell(index);
  }

  onMouseMove(index: number): void {
    if (this.isDrawing) {
      this.paintCell(index);
    }
  }

  onMouseUp(): void {
    this.isDrawing = false;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  /**
   * Paints a single cell by setting its value to 255 (black).
   * Uses a spread copy so OnPush change detection sees a new reference.
   */
  private paintCell(index: number): void {
    if (this.grid[index] === 255) return; // skip re-renders for already-painted cells
    const updated = [...this.grid];
    updated[index] = 255;
    this.grid = updated;
    this.cdr.markForCheck();
  }

  /** Resets every cell to 0 (white). */
  clearGrid(): void {
    this.grid = new Array(784).fill(0);
    this.cdr.markForCheck();
  }

  /**
   * Returns a Float32Array of 784 values normalised to [0, 1].
   * Suitable as MNIST model input with shape [1, 28, 28, 1] or [1, 784].
   *
   * Example usage with TensorFlow.js:
   *   const data  = this.drawWidget.getImageData();
   *   const tensor = tf.tensor4d(data, [1, 28, 28, 1]);
   */
  getImageData(): Float32Array {
    return new Float32Array(this.grid.map(v => v / 255));
  }

  doInference()
  {
    this.m_RestService.doPredictionPixel(this.getImageData()).subscribe(
      (result)=>
      {
        console.log(result);
      }
    );

  }
}
