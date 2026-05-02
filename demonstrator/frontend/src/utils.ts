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

export function float32ArrayToImageUrl(float32Array: Float32Array, width: number, height: number): string {
  // Step 1: Normalize the Float32Array to [0, 255]
  const normalizedArray = new Uint8ClampedArray(float32Array.length*4);
  let j=0
  for (let i = 0; i < float32Array.length; i++) {
    let v:number= Math.min(255, Math.max(0, float32Array[i] * 255)); // Scale to [0, 255]
    normalizedArray[j++] = v; //R
    normalizedArray[j++] = v; //G 
    normalizedArray[j++] = v; //B
    normalizedArray[j++] = 255; //A
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