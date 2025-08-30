import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get error info (type, value, traceback)
    file_name = exc_tb.tb_frame.f_code.co_filename  # Which file caused the error
    line_number = exc_tb.tb_lineno  # Which line number caused the error
    error_message = f"Error occurred in python script [{file_name}] at line [{line_number}] with message [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Call parent class constructor (Exception)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
