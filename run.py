"""Start the local book workspace with Python."""
import multiprocessing
import os
import uvicorn

if __name__ == '__main__':
    multiprocessing.freeze_support()
    uvicorn.run('backend.app:app', host='127.0.0.1', port=int(os.environ.get('PORT', '3000')))
