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

    // Create the request
    let request = tonic::Request::new(RegisterRequest {
        node_id: "node_id_1".to_string()
    });

    // Send the request and get the response
    let response = client.expect("REASON").register_node(request).await;

    // Print the response
    let register_response = response.expect("REASON").into_inner();
    println!("Success: {}", register_response.success);
    println!("Message: {}", register_response.message);
    });



}