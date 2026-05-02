// ai-widget.component.ts
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatStepperModule } from '@angular/material/stepper';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { FormsModule } from '@angular/forms';
import { SentraClientDashboard } from '../sentra-client-dashboard/sentra-client-dashboard';

interface AiStep {
  label: string;
  info: string;
}

@Component({
  selector: 'app-ai-widget',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonToggleModule,
    MatStepperModule,
    MatIconModule,
    MatCardModule,
    MatButtonModule,
  ],
  templateUrl: './ai-widget.component.html',
  styleUrls: ['./ai-widget.component.scss'],
})
export class AiWidgetComponent {
  mode: string = 'normal';
  selectedIndex: number = 0;
  completedSteps: boolean[] = [];

  @Input() m_clientDashboard!: SentraClientDashboard;

  normalSteps: AiStep[] = [
    {
      label: 'Select Host',
      info: 'Choose the remote inference host. The system will ping available endpoints and rank them by latency and capability before presenting the best candidates.',
    },
    {
      label: 'Send Image',
      info: 'Upload or capture an image and transmit it securely to the selected host. Image is compressed and encrypted in transit using TLS 1.3.',
    },
    {
      label: 'Do Inference',
      info: 'The host runs the ML model on the received image. Results—including class labels and confidence scores—are returned and rendered here in real time.',
    },
  ];

  privacySteps: AiStep[] = [
    {
      label: 'Do Attestation',
      info: 'Remote attestation verifies that each compute node is running trusted, unmodified software inside a secure enclave (TEE). Attestation reports are validated before proceeding.',
    },
    {
      label: 'Select Committee',
      info: 'A committee of independent, attested nodes is randomly selected to participate in the computation. This ensures no single party can access the full data.',
    },
    {
      label: 'Send Shares',
      info: 'Your image is split into secret shares using a threshold secret-sharing scheme. Each committee member receives exactly one share—never enough to reconstruct the original.',
    },
    {
      label: 'Do Inference',
      info: 'The committee jointly evaluates the ML model via secure multi-party computation (MPC). The final prediction is revealed only to you, preserving full data privacy.',
    },
  ];

  get currentSteps(): AiStep[] {
    return this.mode === 'normal' ? this.normalSteps : this.privacySteps;
  }

  onModeChange(newMode: string): void {
    this.mode = newMode;
    this.selectedIndex = 0;
    this.completedSteps = [];
  }

  doRemoteAttestation()
  {
    this.m_clientDashboard.doRemoteAttestation();
  }

  doCommitteeSelection()
  {
this.m_clientDashboard.doCommitteeSelection();
  }

    doDistributeShares()
  {
this.m_clientDashboard.doDistributeShares();
  }

    doInference()
  {
this.m_clientDashboard.doInference();
  }

      doImageUpload()
  {
this.m_clientDashboard.doImageUpload();
  }

      doNodeSelection()
  {
this.m_clientDashboard.doNodeSelection();
  }
  
  handleStep(index:number)
  {
    if(this.mode==='privacy')
    {
      switch(index)
      {
        case 0:
          this.doRemoteAttestation();
          break;
        case 1:
          this.doCommitteeSelection();
          break;
        case 2:
          this.doDistributeShares();
          break;
        case 3:
          this.doInference();
          break;
      }
    }
    else//normal mode
    {
        switch(index)
      {
        case 0:
          this.doNodeSelection();
          break;
        case 1:
          this.doImageUpload();
          break;
        case 2:
          this.doInference();
          break;
      }
    }
  }

  completeStep(index: number): void {
    this.handleStep(index);
    this.completedSteps[index] = true;
    // Advance to next step if one exists
    if (index + 1 < this.currentSteps.length) {
      this.selectedIndex = index + 1;
    }
  }

  isStepCompleted(index: number): boolean {
    return !!this.completedSteps[index];
  }
}
