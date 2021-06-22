Extract the zip file. 

Open a terminal in that directory and run the following command:\n
pip install -r requirements.txt.\n
This will load the required packages.\n

The 'mode' can be supplied via command line arguments.
\nmode=0 means no chance constraint
\nmode=1 means gaussian
\nmode=2 means distributionally robust.
\nFor example, to run the code with mode 1, type 'python MLFsim.py 1' in a terminal/command line (without quotes)
\nTo run with mode 0, no argument needs to be supplied, so simply type 'python MLFsim.py', but 'python MLFsim.py 0' would also work.

\nI have hardcoded some ranges, and the output will be shown.
