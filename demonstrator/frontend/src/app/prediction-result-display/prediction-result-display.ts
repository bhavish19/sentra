import { Component, Input, OnInit, SimpleChanges } from '@angular/core';
import type { EChartsCoreOption, ECharts } from 'echarts/core';
import { NgxEchartsDirective, provideEchartsCore } from 'ngx-echarts';
import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import {  BarChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TitleComponent } from 'echarts/components';

echarts.use([BarChart,GridComponent, CanvasRenderer,TitleComponent,LegendComponent]);

@Component({
  selector: 'prediction-result-display',
    imports: [NgxEchartsDirective],

  templateUrl: './prediction-result-display.html',
  styleUrls: ['./prediction-result-display.css'],
    providers: [
    provideEchartsCore({ echarts }),
  ]

})
export class PredictionResultDisplay implements OnInit {
  @Input() predictionProbabilities: number[] = [0.05, 0.05, 0.7, 0.05, 0.1, 0.01, 0.05, 0.1, 0.1, 0.05]; // Array of probabilities for digits 0-9
  @Input() predictedDigit: number = 0; // Predicted digit

  // ECharts options for the bar chart
  chartOptions: EChartsCoreOption = {};

  ngOnInit(): void {
    this.initializeChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['predictionProbabilities']) {
      console.log('Value changed:', changes['predictionProbabilities'].currentValue);
      this.updatePredictionLikelihoods(changes['predictionProbabilities'].currentValue);
    }
  }
  updatePredictionLikelihoods(currentValue:number[]) {
    if(currentValue.length!=10)
      return;
    let series=(this.chartOptions['series'] as any);
    if(series===null)
      return;
    series[0].data =currentValue.reverse();
     this.chartOptions = { ...this.chartOptions };
  }

  // Initialize the chart options
  initializeChart(): void {
    this.chartOptions = {
      title: {
        text: 'Prediction Probabilities',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c}'
      },
      yAxis: {
        type: 'category',
        data: ['9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], // Digits 0-9
        axisLabel: {
          fontSize: 12,
          interval:0
        },
      },
      xAxis: {
        type: 'value',
        min: 0,
        max: 1,
        axisLabel: {
          formatter: '{value}'
        }
      },
      series: [
        {
          data: this.predictionProbabilities, // Probabilities for each digit
          type: 'bar',
          barWidth: '50%',
          itemStyle: {
            color: '#007bff'
          }
        }
      ]
    };
  }
}
