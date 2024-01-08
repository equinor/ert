# server.py
import http.server
import os
import socketserver
import sys

if __name__ == "__main__":
    # Specify the default values
    directory_to_serve = sys.argv[1]
    hostname = sys.argv[2]
    port = int(sys.argv[3])

    # Start the static file server
    os.chdir(directory_to_serve)
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((hostname, port), Handler) as httpd:
        print(f"Serving on port {port} from directory {directory_to_serve}")
        httpd.serve_forever()
