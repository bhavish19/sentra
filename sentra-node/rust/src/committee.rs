use std::{collections::HashMap, fmt, sync::{Arc, RwLock}};
use tonic::transport::{Certificate};
pub struct CommitteeMember
{
    node_id:String,
    grpc_url:String
}

impl fmt::Display for CommitteeMember {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "CommitteeMember: {} -- GRPC URL: {}", self.node_id,self.grpc_url)
    }
}

pub struct Committee
{
    this_node: String, //Node ID of this Sentra node
    ca_cert = Certificate, //the CA certificate for validating inter node TSL connections
    committee: Arc<RwLock<HashMap<String, CommitteeMember>>>
}

impl Default for Committee {
    fn default()->Self
      {
        Committee
        {
            this_node:String::new(),
            ca_cert:Certificate::new(),
            committee:Arc::new(RwLock::new(HashMap::new()))
        }
      }
}

impl fmt::Display for Committee {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let c: std::sync::RwLockReadGuard<'_, HashMap<String, CommitteeMember>>=self.committee.read().unwrap();
        writeln!(f, "Committee with {} members:", c.len());
        for member in c.values()
        {
            write!(f, "\t CommitteeMember: {}", member);
        }
        Ok(())
    }
}

impl Committee
{
    pub fn add(&self,node_id:&String,grpc_url:&String)
    {
        let new_member:CommitteeMember=CommitteeMember
        {
            node_id:node_id.clone(),
            grpc_url:grpc_url.clone()
        };
        self.committee.write().unwrap().insert(new_member.node_id.clone(),new_member);
    }

    pub fn establish_connections(&self)->Result<(),()>
    {
        let c: std::sync::RwLockReadGuard<'_, HashMap<String, CommitteeMember>>=self.committee.read().unwrap();
        for member in c.values()
        {
            if(member.node_id==self.this_node)
                break;
            establish_outgoing_connection(member);
        }
        Ok(())
    }

    pub fn establish_outgoing_connection(&self,sentraNode:&CommitteeMember)
    {
        let rt: tokio::runtime::Runtime = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async
        {
            // Connect to the Sentra Node
            println!("Try to connect to GRPC interface of Sentra node: {} at {}",sentraNode.node_id,sentraNode.grpc_url);
            let mut client: node_message_service_client::NodeMessageServiceClient<tonic::transport::Channel>;
            if !sentra_node.args.use_acme
            {

                let endpoint = Channel::from_shared(sentraNode.grpc_url).expect("REASON-1")
                    .tls_config(
                                tonic::transport::ClientTlsConfig::new()
                                .ca_certificate(self.ca_cert)
                                .domain_name(sentraNode.node_id)
                                ).expect("REASON-2");

                let channel: Channel = endpoint.connect().await.expect("REASON-3");
                client = inter_node_message_service_client::InterNodeMessageServiceClient::new(channel);
            }
            else
            {
                println!("Doing a default connection...");
                client = inter_node_message_service_client::InterNodeMessageServiceClient::connect(sentraNode.grpc_url).await.expect("REASON");
            }

            // Create a channel for sending messages to the node
            println!("Create channels...");
            let (tx, rx) = mpsc::channel(32);

            // Create the stream from the receiver
            println!("Create outbound stream...");
            let outbound: ReceiverStream<NodeMessage> = ReceiverStream::new(rx);

            let tx_register: mpsc::Sender<NodeMessage>=tx.clone();
            println!("Spawn sending hello message thread...");
            let node_id:String=sentraNode.node_id.clone();
            tokio::spawn(async move
                {
                    // Send registration message
                    println!("Sending hello message...");
                    let register_msg: InterNodeMessage = InterNodeMessage {
                        message_type: Some(inter_node_message::MessageType::Hello(HelloMessage {
                            node_id: node_id
                        }))
                    };

                    if tx_register.send(register_msg).await.is_err() {
                        eprintln!("Failed to send hello message");
                        return;
                    }
                });

            tokio::time::sleep(Duration::from_millis(100)).await;

            // Start the bidirectional stream
            println!("Start bidirectional stream...");
            let response_stream: Result<tonic::Response<tonic::Streaming<InterNodeMessage>>, tonic::Status>
                 = client.node_stream(outbound).await;
            println!("Wait for inbound...");

            let mut inbound: tonic::Streaming<InterNodeMessage> = response_stream.expect("REASON").into_inner();

            // Receive messages from the server
            println!("Listening for Node messages...");
            loop
                {
                    match inbound.message().await
                        {
                            Ok(Some(node_msg)) =>
                                {
//                                    sentra_node.handle_server_message(server_msg,tx.clone(),sentra_node.args.fake_attestation);
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
    }
}