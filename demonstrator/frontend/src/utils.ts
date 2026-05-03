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