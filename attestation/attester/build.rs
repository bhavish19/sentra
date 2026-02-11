fn main() -> Result<(), Box<dyn std::error::Error>> {
tonic_build::configure()
        .build_server(false)  // We only need the client    
	.compile(&["../../demonstrator/backend/SentraBackend-GRPC-Services.proto"],&["../../demonstrator/backend"])?;
    Ok(())
}
