#!python
import logging
from suds.client import Client

client = Client("http://www.onvif.org/ver10/device/wsdl")
print client

logging.basicConfig(level=logging.INFO)
logging.getLogger('suds.xsd.schema').setLevel(logging.DEBUG)

url = "http://www.onvif.org/ver10/device/wsdl"
client = Client(url)
a = client.messages.items()
list_of_methods = [method for method in client.wsdl.services[0].ports[0].methods]

print list_of_methods.__len__()


print("fine programma oyy")











