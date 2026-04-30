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