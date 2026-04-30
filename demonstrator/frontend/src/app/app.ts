import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
public staticHeadline: string="Privacy-preserving Machine Learning";
public static staticSubTitle: string="Client Side";

get getSubTitle()
  {
    return App.staticSubTitle;
  }
}
