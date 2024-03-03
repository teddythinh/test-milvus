from pymilvus import connections, utility

connections.connect('default', host='localhost', port='19530')

print(utility.list_collections())