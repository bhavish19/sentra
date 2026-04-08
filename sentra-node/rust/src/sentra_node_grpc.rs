use tonic::{transport::Server,Request,Response,Status};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

tonic::include_proto!("sentra_inter_node_grpc_services");

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
        // Return the receiver stream as the response
        Ok(Response::new(ReceiverStream::new(rx)))
	}
} 

async fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse().unwrap();
    let intern_node_grpc_service = SentraInterNodeMessageService::default();

    println!("Sentra inter node grpc service listening on {}", addr);

    Server::builder()
        .add_service(inter_node_message_service_server::InterNodeMessageServiceServer::new(intern_node_grpc_service))
        .serve(addr)
        .await?;

    Ok(())
}

pub fn startGRPCServer()
{
    let server_thread = std::thread::spawn(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            if let Err(e) = run_server().await {
                eprintln!("Server error: {}", e);
            }
        });
    });
}