import os
from datetime import datetime
from pytz import timezone
from typing import Any

def get_curr_time() -> str:
    """Get current date and time in Shanghai timezone as str."""
    return datetime.now().astimezone(
            timezone('Asia/Shanghai')).strftime("%d/%m/%Y %H:%M:%S")


class Logger: 
    """Class to write message to both output_dir/filename.txt and terminal."""
    def __init__(self, output_dir: str=None, filename: str=None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool=True) -> None:
        "write the message"
        # message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'): 
                message = '\n'+get_curr_time()+' >> '+message[1:]
            else:
                message = get_curr_time()+' >> '+message
        print(message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message+'\n')
                f.flush()  # 立即刷新到磁盘