from datetime import datetime
import time

print("datetime.now():", datetime.now())
print("time.time():", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))