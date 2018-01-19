#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:59:43 2018

References:
    1. https://daanlenaerts.com/blog/2015/06/03/create-a-simple-http-server-with-python-3/
    

@author: weiya
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess

class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    
    # GET
    def do_GET(self):    
        
        #
        print(self.requestline)
        req = self.requestline.split()
        if (req[1] == '/deploy'):
            update()
        # response status code
        self.send_response(200)
        
        # header
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # message
        message = 'hello world!'
        
        self.wfile.write(bytes(message, "utf8"))
        return
    
def update():
    PATH = '../site'
    cmd = 'cd ' + PATH + ' && ' + 'git pull'
    out = subprocess.getoutput(cmd)
    print(out)

def run():
    print('starting server ...')
    
    server_address = ('', 8088)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server ...')
    httpd.serve_forever()
    
    
if __name__ == "__main__":
    run()
        