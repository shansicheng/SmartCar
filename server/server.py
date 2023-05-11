import io
from threading import Thread,Condition
import logging
import logging.handlers
import datetime
import socketserver
from threading import Condition
from http import server


from config import get_config
from camera import Camera

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    global output
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            PAGE = self.set_page(640,480)
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()
    def set_page(self,width,height):
        content = "<img src=\"stream.mjpg\" " + "width=\"{}\" ".format(width) + "height=\"{}\"".format(height) + ">"
        PAGE = "<!DOCTYPE html>\
                <html>\
                <head>\
                <title>Camera</title>\
                </head>\
                <body>\
                <center><h1>The camera outputs:</h1></center>\
                <center>" + content + "</center>\
                </body>\
                </html>"
        return PAGE
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

output : StreamingOutput = StreamingOutput()

#设置日志
# 日期格式
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

# 添加日志器的名称标识
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

all_handler = logging.handlers.TimedRotatingFileHandler(
    filename='all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0)
)
all_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

err_handler = logging.FileHandler('error.log')
err_handler.setLevel(logging.ERROR)
# 格式器
err_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d -%(pathname)s \n%(message)s")
)

# 给logger 添加处理器
logger.addHandler(all_handler)
logger.addHandler(err_handler)

config = get_config()
try:
    logger.info("Begin the server!")
    output = StreamingOutput()
    with Camera(output, config,True) as camera:
        address = (config.server_ip, config.server_port)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
except KeyboardInterrupt:
    logger.info("Stop the server by user!")

