/**
 * Creates the correct WebSocket URL for a given path. This mainly covers the question if the WebSocket request shoudl be made in a secure way (wss) or plain (ws).
 * This is determined based on property of window.location.protocol, i.e. if the app is served by http or https.
 */
export function createWebSocketURLForPath(path:string):string
{
  const proto:string=window.location.protocol;
  console.log("createWebSocketURLForPath() -- base protocol is: "+proto);
  if(proto==="https:")
    return "wss://"+window.location.host+path;
  else
    return "ws://"+window.location.host+path;
}

export function sleep  (ms: number):Promise<void>
  {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
};

export function generateRandomFloat32Array(size:number):Float32Array
{
  const randomArray = new Float32Array(size);
  for (let i = 0; i < randomArray.length; i++) {
    randomArray[i] = Math.random(); // Random number between 0 and 1
  }
  return randomArray;
}

//The colIndex says, which colorisation should be used. 
// colIndeX==0 means gray scale...
// colIndex==1 means read
// colIndex==2 means yellow
// colIndex==3 means blue
// colIndex==4 means green
// colIndex==5 means pink
export function float32ArrayToImageUrl(float32Array: Float32Array, width: number, height: number,colIndex:number): string {
  // Step 1: Normalize the Float32Array to [0, 255]
  const normalizedArray = new Uint8ClampedArray(float32Array.length*4);
  let j=0
  if(colIndex>5)
    colIndex=0;
  for (let i = 0; i < float32Array.length; i++) 
    {
      let v:number= Math.min(255, Math.max(0, float32Array[i] * 255)); // Scale to [0, 255]
      if(colIndex===0||colIndex===1||colIndex===5)
        normalizedArray[j++] = v; //R
      else
        normalizedArray[j++] = 0; //R
      if(colIndex===0||colIndex===2||colIndex===4)
        normalizedArray[j++] = v; //G 
      else
        normalizedArray[j++] = 0; //G
      if(colIndex===0||colIndex===3||colIndex===4||colIndex===5)
        normalizedArray[j++] = v; //B
      else
        normalizedArray[j++] = 0; //B
      normalizedArray[j++] = 255; //A - opaque
  }

  // Step 2: Create an ImageData object
  const imageData = new ImageData(normalizedArray, width, height);

  // Step 3: Draw the ImageData on a Canvas
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get 2D context');
  }
  ctx.putImageData(imageData, 0, 0);

  // Step 4: Export the Canvas as a Data URL
  return canvas.toDataURL('image/png'); // Returns a Base64-encoded PNG image URL
}

export function generateBase64Image(digit:number):string|null {
    if (typeof digit !== 'number' || digit < 0 || digit > 9) 
      {
        throw new Error("Input must be a single digit (0-9).");
      }

    // Create a canvas element
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if(ctx===null)
      return null;

    // Set canvas dimensions
    canvas.width = 56; // Width of the image
    canvas.height = 56; // Height of the image

    // Set background color (optional)
    ctx.fillStyle = "#ffffff"; // White background
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Set text properties
    ctx.font = "bold 30px Arial"; // Font size and style
    ctx.fillStyle = "#000000"; // Black text color
    ctx.textAlign = "center"; // Center align text
    ctx.textBaseline = "middle"; // Middle align text

    // Draw the digit in the center of the canvas
    ctx.fillText(digit.toString(), canvas.width / 2, canvas.height / 2);

    // Convert the canvas to a Base64 image URL
    const base64Image:string = canvas.toDataURL("image/png");

    return base64Image;
}

export function generateFlashingArrowSVG(width:number,height:number):string
{
  let svg=`<?xml version='1.0' encoding='UTF-8'?> \
<svg xmlns='http://www.w3.org/2000/svg' width="`+width+`" height="500" viewBox="0 0 `+width+` 500"> \
  <defs> \
    <filter id='neon-glow' x='-80%' y='-80%' width='260%' height='260%'>
      <feGaussianBlur in='SourceGraphic' stdDeviation='9' result='blur_fat'/>
      <feGaussianBlur in='SourceGraphic' stdDeviation='4' result='blur_med'/>
      <feGaussianBlur in='SourceGraphic' stdDeviation='2' result='blur_fine'/>
      <feMerge>
        <feMergeNode in='blur_fat'/>
        <feMergeNode in='blur_fat'/>
        <feMergeNode in='blur_med'/>
        <feMergeNode in='blur_fine'/>
        <feMergeNode in='SourceGraphic'/>
      </feMerge>
    </filter>
  </defs>
`
  let posX:number=10;
  let posY:number=65;
  let curTim:number=0.0;
  let dT:number=0.13;
  let xA:number=(width-posX)/55;
  let yA:number=(height-posY)/40;
  let duration:number=(xA+yA)*dT+2;
  while(posX<width-50)
  {
    let s=`
      <g filter="url(#neon-glow)">
    <polyline points="`+posX+`,15 `+(posX+22)+`,35 `+posX+`,55" fill="none" stroke="#0088bb" stroke-width="14" stroke-linecap="round"
     stroke-linejoin="round" opacity="0.12">
      <animate attributeName="opacity" values="0.12;0.12;0.50;0.12;0.12" keyTimes="0;0.12;0.25;0.38;1" dur="`+duration+`s" begin="`+curTim+`s" 
      repeatCount="indefinite" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1"/>
    </polyline>
    <polyline points="`+posX+`,15 `+(posX+22)+`,35 `+posX+`,55" fill="none" stroke="#00e5ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="0.12">
      <animate attributeName="opacity" values="0.12;0.12;1.0;0.12;0.12" keyTimes="0;0.12;0.25;0.38;1" dur="`+duration+`s" begin="`+curTim+`s" repeatCount="indefinite" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1"/>
    </polyline>
  </g>`;
  svg+=s;
  posX+=55;
  curTim+=dT;
  }
  posX-=45;
   while(posY<height)
  { 
  svg+=`
  <g filter="url(#neon-glow)">
    <polyline points="`+posX+","+posY+" "+(posX+20)+","+(posY+20)+" "+(posX+40)+" "+posY+`" fill="none" stroke="#0088bb" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" opacity="0.12">
      <animate attributeName="opacity" values="0.12;0.12;0.50;0.12;0.12" keyTimes="0;0.12;0.25;0.38;1" dur="`+duration+`s" begin="`+curTim+`s" repeatCount="indefinite" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1"/>
    </polyline>
    <polyline points="`+posX+","+posY+" "+(posX+20)+","+(posY+20)+" "+(posX+40)+" "+posY+`" fill="none" stroke="#00e5ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="0.12">
      <animate attributeName="opacity" values="0.12;0.12;1.0;0.12;0.12" keyTimes="0;0.12;0.25;0.38;1" dur="`+duration+`s" begin="`+curTim+`s" repeatCount="indefinite" calcMode="spline" keySplines="0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1"/>
    </polyline>
  </g>`
  posY+=40;
  curTim+=dT;
  }

  svg+="</svg>";
  return svg;
}