mod sentra_attester;
mod sentra_rest_server;

fn main()
{
    sentra_rest_server::start_rest_server();
}