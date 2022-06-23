
To compile from root-dir

python -m grpc_tools.protoc  -I ert_shared/ensemble_evaluator/experiment_server --python_out=. --grpc_python_out=. ert_shared/ensemble_evaluator/experiment_server/experimentserver.proto 

Compile from local-dir
python -m grpc_tools.protoc  -I . --python_out=. --grpc_python_out=. experimentserver.proto 
