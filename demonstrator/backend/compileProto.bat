python3.14 -m grpc_tools.protoc -I. --python_out=./sentrabackend/grpc/generated --grpc_python_out=./sentrabackend/grpc/generated ./SentraBackend-GRPC-Services.proto
python3.14 -m PythonSed.sed -e "s/import SentraBackend/from . import SentraBackend/" ./sentrabackend/grpc/generated/SentraBackend_GRPC_Services_pb2_grpc.py > ./sentrabackend/grpc/generated/out.py
move .\sentrabackend\grpc\generated\out.py .\sentrabackend\grpc\generated\SentraBackend_GRPC_Services_pb2_grpc.py
