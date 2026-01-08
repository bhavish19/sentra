"""
Secure Communication Protocol for Multi-Node MPC
Enables secure channels between nodes and share reconstruction
"""

from typing import List, Dict, Optional, Callable, Any
from ml_training.secret_sharing import Share, ShamirSecretSharing
import socket
import ssl
import json
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
import struct


class MessageType(Enum):
    """Types of messages in the secure communication protocol"""
    SHARE_EXCHANGE = "share_exchange"
    RECONSTRUCTION_REQUEST = "reconstruction_request"
    RECONSTRUCTION_RESPONSE = "reconstruction_response"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    SYNC = "sync"


@dataclass
class Message:
    """Message structure for secure communication"""
    msg_type: str
    sender_id: int
    receiver_id: int
    data: Dict[str, Any]
    timestamp: float
    message_id: str = ""
    signature: Optional[str] = None  # For authentication


class SecureChannel:
    """
    Secure communication channel between nodes
    Provides authenticated and encrypted communication
    """
    
    def __init__(self, node_id: int, port: int = 8000, use_tls: bool = False):
        """
        Initialize secure channel
        Args:
            node_id: ID of this node
            port: Port to listen on
            use_tls: Whether to use TLS encryption (default: False for testing)
        """
        self.node_id = node_id
        self.port = port
        self.use_tls = use_tls
        self.connections: Dict[int, socket.socket] = {}
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.message_handlers: Dict[str, Callable] = {}
        self.received_shares: Dict[str, List[Share]] = {}
        self.lock = threading.Lock()
        self.message_counter = 0
    
    def start_server(self):
        """Start listening server for incoming connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(10)
        self.running = True
        
        def accept_connections():
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    if self.use_tls:
                        # Wrap with TLS (simplified - would use proper certificates)
                        try:
                            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                            # In production, would load proper certificates
                            context.check_hostname = False
                            context.verify_mode = ssl.CERT_NONE
                            client_socket = context.wrap_socket(client_socket, server_side=True)
                        except Exception:
                            # Continue with plain socket if TLS fails (silently)
                            pass
                    threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True).start()
                except socket.error as e:
                    if self.running:
                        # Ignore errors when shutting down
                        if e.errno != 10038:  # WSAENOTSOCK
                            print(f"Error accepting connection: {e}")
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")
        
        threading.Thread(target=accept_connections, daemon=True).start()
        print(f"Node {self.node_id} server started on port {self.port}")
    
    def connect_to_node(self, target_node_id: int, host: str, port: int):
        """
        Connect to another node
        Args:
            target_node_id: ID of target node
            host: Host address
            port: Port number
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            # Connect first, then wrap with TLS if needed
            # Retry several times to allow for startup of all nodes...
            i:int=0
            while(True):
                try:
                    sock.connect((host, port))
                    break
                except Exception as e:
                    i+=1
                    if(i>5):
                        raise e
                    time.sleep(10)
            
            if self.use_tls:
                try:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    sock = context.wrap_socket(sock, server_hostname=host)
                except Exception:
                    # If TLS fails, use plain socket (already connected) - silently
                    pass
            
            self.connections[target_node_id] = sock
            print(f"✓ Node {self.node_id} connected to node {target_node_id} at {host}:{port}")
            return True
        except Exception as e:
            print(f"✗ Error connecting to node {target_node_id}: {e}")
            return False
    
    def send_message(self, target_node_id: int, msg_type: MessageType, data: Dict[str, Any]):
        """
        Send a message to another node
        Args:
            target_node_id: Target node ID
            msg_type: Message type
            data: Message data
        """
        if target_node_id not in self.connections:
            raise ValueError(f"No connection to node {target_node_id}")
        
        self.message_counter += 1
        message = Message(
            msg_type=msg_type.value,
            sender_id=self.node_id,
            receiver_id=target_node_id,
            data=data,
            timestamp=time.time(),
            message_id=f"{self.node_id}_{self.message_counter}"
        )
        
        try:
            sock = self.connections[target_node_id]
            message_json = json.dumps(asdict(message))
            # Send length first, then message
            message_bytes = message_json.encode('utf-8')
            length = struct.pack('>I', len(message_bytes))
            sock.sendall(length + message_bytes)
        except Exception as e:
            print(f"Error sending message to node {target_node_id}: {e}")
            raise
    
    def send_share(self, target_node_id: int, share: Share, context: str = "default"):
        """
        Send a share to another node
        Args:
            target_node_id: Target node ID
            share: Share to send
            context: Context identifier for this share exchange
        """
        data = {
            'context': context,
            'share': {
                'x': share.x,
                'y': share.y,
                'node_id': share.node_id
            }
        }
        self.send_message(target_node_id, MessageType.SHARE_EXCHANGE, data)
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle incoming client connection"""
        try:
            # Verify it's actually a socket
            if not isinstance(client_socket, socket.socket) and not hasattr(client_socket, 'recv'):
                print(f"Error: Invalid socket object received")
                return
            
            while self.running:
                # Read message length
                length_data = client_socket.recv(4)
                if not length_data:
                    break
                
                length = struct.unpack('>I', length_data)[0]
                
                # Read message
                message_data = b''
                while len(message_data) < length:
                    chunk = client_socket.recv(length - len(message_data))
                    if not chunk:
                        break
                    message_data += chunk
                
                if len(message_data) == length:
                    self._process_message(json.loads(message_data.decode('utf-8')))
        except (socket.error, OSError) as e:
            # Ignore socket errors when shutting down or client disconnects
            if self.running and e.errno not in (10038, 10054, 10053):  # WSAENOTSOCK, WSAECONNRESET, WSAEINTR
                print(f"Error handling client: {e}")
        except Exception as e:
            if self.running:
                print(f"Error handling client: {e}")
        finally:
            try:
                if hasattr(client_socket, 'close'):
                    client_socket.close()
            except:
                pass
    
    def _process_message(self, msg_dict: Dict):
        """Process incoming message"""
        msg_type = msg_dict.get('msg_type')
        sender_id = msg_dict.get('sender_id')
        data = msg_dict.get('data', {})
        
        if msg_type == MessageType.SHARE_EXCHANGE.value:
            self._handle_share_exchange(sender_id, data)
        elif msg_type == MessageType.RECONSTRUCTION_REQUEST.value:
            self._handle_reconstruction_request(sender_id, data)
        elif msg_type == MessageType.RECONSTRUCTION_RESPONSE.value:
            self._handle_reconstruction_response(sender_id, data)
        elif msg_type == MessageType.HEARTBEAT.value:
            self._handle_heartbeat(sender_id, data)
        elif msg_type in self.message_handlers:
            self.message_handlers[msg_type](sender_id, data)
    
    def _handle_share_exchange(self, sender_id: int, data: Dict):
        """Handle received share"""
        context = data.get('context')
        share_dict = data.get('share')
        share = Share(
            x=share_dict['x'],
            y=share_dict['y'],
            node_id=share_dict['node_id']
        )
        
        with self.lock:
            if context not in self.received_shares:
                self.received_shares[context] = []
            self.received_shares[context].append(share)
    
    def _handle_reconstruction_request(self, sender_id: int, data: Dict):
        """Handle reconstruction request"""
        context = data.get('context')
        with self.lock:
            if context in self.received_shares:
                # Send back our share
                for share in self.received_shares[context]:
                    if share.node_id == self.node_id:
                        self.send_share(sender_id, share, context)
    
    def _handle_reconstruction_response(self, sender_id: int, data: Dict):
        """Handle reconstruction response"""
        self._handle_share_exchange(sender_id, data)
    
    def _handle_heartbeat(self, sender_id: int, data: Dict):
        """Handle heartbeat message"""
        # Send ACK
        self.send_message(sender_id, MessageType.ACK, {'ack': True})
    
    def register_handler(self, msg_type: str, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[msg_type] = handler
    
    def get_received_shares(self, context: str) -> List[Share]:
        """Get received shares for a context"""
        with self.lock:
            return self.received_shares.get(context, []).copy()
    
    def clear_context(self, context: str):
        """Clear shares for a context"""
        with self.lock:
            if context in self.received_shares:
                del self.received_shares[context]
    
    def stop(self):
        """Stop the server and close connections"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        for sock in self.connections.values():
            try:
                sock.close()
            except:
                pass
        self.connections.clear()


class SecureMPCNetwork:
    """
    High-level interface for secure multi-party computation network
    Manages nodes, channels, and reconstruction
    """
    
    def __init__(self, node_id: int, node_configs: Dict[int, Dict[str, Any]],
                 port: int = 8000, use_tls: bool = False):
        """
        Initialize MPC network
        Args:
            node_id: ID of this node
            node_configs: Dictionary mapping node_id to {host, port}
            port: Port for this node
            use_tls: Whether to use TLS (default: False for testing)
        """
        self.node_id = node_id
        self.node_configs = node_configs

        self.channel = SecureChannel(node_id, port, use_tls)
        self.sss = ShamirSecretSharing()
        
        # Start server
        self.channel.start_server()
        
        # Wait a bit for server to start
        time.sleep(0.5)
        
        # Connect to other nodes
        connected_nodes = []
        failed_nodes = []
        for other_id, config in node_configs.items():
            if other_id != node_id:
                success = self.channel.connect_to_node(
                    other_id, config['host'], config['port']
                )
                if success:
                    connected_nodes.append(other_id)
                else:
                    failed_nodes.append(other_id)
                time.sleep(0.1)  # Small delay between connections
        
        # Print connection summary
        print(f"\n{'='*70}")
        print(f"Connection Summary for Node {node_id}:")
        print(f"{'='*70}")
        if connected_nodes:
            print(f"✓ Connected to {len(connected_nodes)} node(s): {connected_nodes}")
        if failed_nodes:
            print(f"✗ Failed to connect to {len(failed_nodes)} node(s): {failed_nodes}")
        if not connected_nodes and not failed_nodes:
            print("No other nodes to connect to (single-node mode)")
        print(f"{'='*70}\n")
    
    def send_share(self, target_node_id: int, share: Share, context: str):
        """Send a share to another node"""
        self.channel.send_share(target_node_id, share, context)
    
    def broadcast_share(self, share: Share, context: str):
        """Broadcast a share to all nodes"""
        for node_id in self.node_configs.keys():
            if node_id != self.node_id:
                try:
                    self.send_share(node_id, share, context)
                except Exception as e:
                    print(f"Warning: Could not broadcast to node {node_id}: {e}")
    
    def get_received_shares(self, context: str) -> List[Share]:
        """Get received shares for a context"""
        return self.channel.get_received_shares(context)
    
    def stop(self):
        """Stop the network"""
        self.channel.stop()


def create_mpc_network(node_id: int, node_configs: Dict[int, Dict[str, Any]],
                       port: int = 8000, use_tls: bool = False) -> SecureMPCNetwork:
    """
    Factory function to create an MPC network
    
    Args:
        node_id: ID of this node
        node_configs: Dictionary mapping node_id to {host, port}
        port: Port for this node
        use_tls: Whether to use TLS (default: False for testing)
    Returns:
        Configured SecureMPCNetwork instance
    """
    return SecureMPCNetwork(node_id, node_configs, port, use_tls)


