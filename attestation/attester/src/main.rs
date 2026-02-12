mod sentra_attester;
//mod sentra_rest_server;

// Include the generated code from the proto file
tonic::include_proto!("sentra_backend_grpc_services");


//use sentra_backend_grpc_services::node_registration_client::NodeRegistrationClient;
//use RegisterRequest;


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

  let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {

   // Connect to the server
//    let client = node_registration_client::NodeRegistrationClient::connect("http://sentra-backend:8000").await;
    println!("Try to connect to: {}",server_address);
    let client = node_registration_client::NodeRegistrationClient::connect(server_address).await;

    // Create a channel for sending messages to the server
    let (tx, rx) = mpsc::channel(32);
    
    // Create the stream from the receiver
    let outbound = ReceiverStream::new(rx);
    
    // Start the bidirectional stream
    let response_stream = client.node_stream(outbound).await;
    let mut inbound = response_stream.into_inner();

tokio::spawn(async move {
        // Send registration message
        println!("Sending registration...");
        let register_msg = NodeMessage {
            node_id: node_id_sender.clone(),
            message_type: Some(node_message::MessageType::Register(RegisterMessage {
                node_id: node_id_sender.clone()
            })),
        };
        
        if tx.send(register_msg).await.is_err() {
            eprintln!("Failed to send registration");
            return;
        }
        
    });
    
    // Receive messages from the server
    println!("Listening for server messages...");
    while let Some(server_msg) = inbound.message().await {
        match server_msg.message_type {
            Some(server_message::MessageType::Response(ack)) => {
                println!("✓ ACK: {} - {}", ack.success, ack.message);
            }
            None => {
                println!("Received empty message");
            }
        }
    }

    println!("Stream closed by server");
    Ok(())


    });



}