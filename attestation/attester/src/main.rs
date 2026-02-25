mod sentra_attester;
mod acme;

use std::{error::Error, time::Duration};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, Certificate};
use hostname;
use rustls::crypto::{aws_lc_rs, CryptoProvider};
use rustc_version_runtime;
// Include the generated code from the proto file
tonic::include_proto!("sentra_backend_grpc_services");

mod command_line_options;

const SENTRA_NODE_VERSION: &str = "00.03.078";

let args:command_line_options::CommandLineOptions= argh::from_env();

fn main()
{
    print!("Starting Sentra Node version: {} [compile using {:?}]",SENTRA_NODE_VERSION,rustc_version_runtime::version());
    
    CryptoProvider::install_default(aws_lc_rs::default_provider()).expect("Failed to install crypto provider");
    let mut grpc_cert:Option<String>=None;
    let mut grpc_key:Option<String>=None;
    if args.use_acme
    {
        match acme::get_tls_certificate(&args.acme_url,&args.acme_cert)
            {
                Ok((cert, key)) => 
                    {
                        println!("Got certificate!");
                        grpc_cert=Some(cert);
                        grpc_key=Some(key);
                    }
                Err(e) => 
                    {
                        eprintln!("MAIN ERROR: {}", e);
                        eprintln!("Full error chain:");
                        let mut current: Option<&dyn Error> = e.source();
                        while let Some(cause) = current 
                            {
                                eprintln!("Caused by: {}", cause);
                                current = cause.source();
                            }
                        return;
                    }
        };
    }
    let grp_server_url: String=args.grpc_url;
     
    let node_id: String=match hostname::get() 
        {
            Ok(name) => 
                {
                    let hname: String=name.to_string_lossy().to_string();
                    println!("Hostname: {}", hname);
                    hname
                }
            Err(e) => 
                {
                    eprintln!("Failed to get hostname: {}", e);
                    String::from("Unknown")
                }
        };

    let rt: tokio::runtime::Runtime = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async 
        {
            // Connect to the server
            println!("Try to connect to GRPC interface of Sentra backend: {}",grp_server_url);
            let mut client: node_message_service_client::NodeMessageServiceClient<tonic::transport::Channel>;
            if args.use_acme
            {
                let ca_cert = Certificate::from_pem(grpc_cert.unwrap().as_bytes());
                println!("Loaded CA certificate");

                let endpoint = Channel::from_shared(grp_server_url).expect("REASON-1")
                    .tls_config(            
                                tonic::transport::ClientTlsConfig::new()
                                .ca_certificate(ca_cert)
                                .domain_name("sentra-backend")
                                ).expect("REASON-2");

                let channel: Channel = endpoint.connect().await.expect("REASON-3");
                client = node_message_service_client::NodeMessageServiceClient::new(channel);
            }
            else
            {
                println!("Doing a default connection...");
                client = node_message_service_client::NodeMessageServiceClient::connect(grp_server_url).await.expect("REASON");
            }

            // Create a channel for sending messages to the server
            println!("Create channels...");
            let (tx, rx) = mpsc::channel(32);
    
            // Create the stream from the receiver
            println!("Create outbound stream...");
            let outbound: ReceiverStream<NodeMessage> = ReceiverStream::new(rx);
    
            let tx_register: mpsc::Sender<NodeMessage>=tx.clone();
            println!("Spawn registration thread...");
            tokio::spawn(async move 
                {
                    // Send registration message
                    println!("Sending registration...");
                    let register_msg: NodeMessage = NodeMessage {
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
            let response_stream: Result<tonic::Response<tonic::Streaming<ServerMessage>>, tonic::Status>
                 = client.node_stream(outbound).await;
            println!("Wait for inbound...");

            let mut inbound: tonic::Streaming<ServerMessage> = response_stream.expect("REASON").into_inner();
    
            // Receive messages from the server
            println!("Listening for server messages...");
            loop 
                {
                    match inbound.message().await
                        {
                            Ok(Some(server_msg)) => 
                                {
                                    handle_server_message(server_msg,tx.clone());
                                }
                            Ok(None) =>
                                {
                                    // Stream ended
                                    println!("Stream closed by server");
                                    break;
                                }
                            Err(status) => 
                                {
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
            let tx_clone: mpsc::Sender<NodeMessage>=tx;
	        tokio::spawn(async move {
                // Send attestatin message
                let quote: Vec<u8>=sentra_attester::generate_attestation_report(args.fake_attestation);
                println!("Sending attestation...");
                let register_msg: NodeMessage = NodeMessage {
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
        Some(server_message::MessageType::JoinCommitteeRequest(request)) => {
            println!("Received JoinCommittee request -- Committee: {:?}", request.committee);
        },
        None => {
            println!("Received empty message");
        }
    }

}