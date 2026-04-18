use tonic::{transport::Server,Request,Response,Status};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

tonic::include_proto!("sentra_inter_node_grpc_services");

use main::SentraNode;

#[derive(Default)]
pub struct SentraInterNodeMessageService{}

#[tonic::async_trait]
impl inter_node_message_service_server::InterNodeMessageService for SentraInterNodeMessageService
{
    type InterNodeStreamStream = ReceiverStream<Result<InterNodeMessage, Status>>;

    async fn inter_node_stream(&self,request: Request<tonic::Streaming<InterNodeMessage>>) -> Result<Response<Self::InterNodeStreamStream>, Status> 
    {
        println!("New bidirectional inter node meassage stream started! - We are the server");
        let mut incoming_stream = request.into_inner();

            // Create a channel to send messages back to the client
        let (tx, rx) = mpsc::channel(32);


        // Spawn a task to handle incoming messages
        tokio::spawn(async move {
            while let result = incoming_stream.message().await {
                match result {
                    Ok(Some(message)) => 
                    {
                         handle_node_message("test",message)
                    }
                     Ok(None) =>
                                {
                                    // Stream ended
                                    println!("Stream closed by node");
                                    break;
                                }
                    Err(e) => {
                        eprintln!("Error receiving message: {}", e);
                        break;
                    }
                }
            }
            println!("Client disconnected or stream ended");
        });

        // Return the receiver stream as the response
        Ok(Response::new(ReceiverStream::new(rx)))
	}
} 

pub fn handle_node_message(this_node:String,node_msg: InterNodeMessage)
    {
        match node_msg.message_type 
        {
            Some(inter_node_message::MessageType::Hello(hello)) => 
            {
                println!("Received Hello-Message from {} to {}...",this_node,hello.node_id);
	    },
            Some(inter_node_message::MessageType::Heartbeat(heartbeat)) => 
            {
                println!("Received Heartbeat Message...");
	    },
	    None => {
        	println!("Received empty inter node message");
    	    }
        }
    }

async fn run_server(sentra_node:&SentraNode) -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse().unwrap();
    let intern_node_grpc_service = SentraInterNodeMessageService::default();

    println!("Sentra inter node grpc service listening on {}", addr);

    Server::builder()
        .add_service(inter_node_message_service_server::InterNodeMessageServiceServer::new(intern_node_grpc_service))
        .serve(addr)
        .await?;

    Ok(())
}

pub fn startGRPCServer(sentra_node:&SentraNode)
{
    let server_thread = std::thread::spawn(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            if let Err(e) = run_server(sentra_node).await {
                eprintln!("Server error: {}", e);
            }
        });
    });
}