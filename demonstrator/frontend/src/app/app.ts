import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { RestService } from '../rest.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
public staticHeadline: string="Privacy-preserving Machine Learning";
public static staticSubTitle: string="Client Side";

constructor(private m_RestService: RestService)
{
  
}

get getSubTitle()
  {
    return App.staticSubTitle;
  }

onBILogoClicked() 
{
  this.m_RestService.doReset().subscribe();
}


}
