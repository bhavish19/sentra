mod sentra_attester;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::time::Duration;
use hostname;

// Include the generated code from the proto file
tonic::include_proto!("sentra_backend_grpc_services");

fn main()
{


let args: Vec<String> = std::env::args().collect();
    let server_address = if args.len() > 1 {
        args[1].clone()
    } else {
        "http://127.0.0.1:8000".to_string()
    };

/*  let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        sentra_rest_server::start_rest_server().await;
    });
*/


let node_id=match hostname::get() {
        Ok(name) => {
             let hname=name.to_string_lossy().to_string();
            println!("Hostname: {}", hname);
            hname
        }
        Err(e) => {
            eprintln!("Failed to get hostname: {}", e);
	    String::from("Unknown")
        }
    };

  let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {

   // Connect to the server
    println!("Try to connect to: {}",server_address);
    let client = node_message_service_client::NodeMessageServiceClient::connect(server_address).await;

    // Create a channel for sending messages to the server
    println!("Create channels...");
    let (tx, rx) = mpsc::channel(32);
    
    // Create the stream from the receiver
    println!("Create outbound stream...");
    let outbound = ReceiverStream::new(rx);
    
    let tx_register=tx.clone();
    println!("Spawn registration thread...");
    tokio::spawn(async move {
        // Send registration message
        println!("Sending registration...");
        let register_msg = NodeMessage {
            message_type: Some(node_message::MessageType::Register(RegisterRequest {
                node_id: node_id.clone()
            }))
        };
        
        if tx_register.send(register_msg).await.is_err() {
            eprintln!("Failed to send registration");
            return;
        }

        
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Start the bidirectional stream
    println!("Start bidirectional stream...");
    let response_stream = client.expect("REASON").node_stream(outbound).await;
    println!("Wait for inbound...");

    let mut inbound = response_stream.expect("REASON").into_inner();
    
    // Receive messages from the server
    println!("Listening for server messages...");
    loop {
        match inbound.message().await {
            Ok(Some(server_msg)) => {
                handle_server_message(server_msg,tx.clone());
            }
            Ok(None) => {
                // Stream ended
                println!("Stream closed by server");
                break;
            }
            Err(status) => {
                eprintln!("Error receiving message: {}", status);
                break;
            }
        }
    }
    println!("Exiting....");


    });
}
fn handle_server_message(server_msg: ServerMessage,tx:mpsc::Sender<NodeMessage>) {
    match server_msg.message_type {
        Some(server_message::MessageType::Response(ack)) => {
            println!("✓ ACK: {} - {}", ack.success, ack.message);
        },
        Some(server_message::MessageType::Attestation(req)) => {
            println!("Attestation request with nonce: {}...",req.nonce);        
            println!("Spwan send quote thread...");
            let tx_clone=tx;
	        tokio::spawn(async move {
                // Send attestatin message
                let quote=sentra_attester::generate_attestation_report();
                println!("Sending attestation...");
                let register_msg = NodeMessage {
                    message_type: Some(node_message::MessageType::Quote(AttestationResponse {
                        report: quote
                    }))
                };
                
                if tx_clone.send(register_msg).await.is_err() {
                    eprintln!("Failed to send attestation");
                    return;
                }
            });
        },
        None => {
            println!("Received empty message");
        }
    }

}