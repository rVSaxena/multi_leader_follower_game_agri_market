Download the files. 

Open a terminal in that directory and run the following command:

pip install -r requirements.txt.

This will load the required packages.

The 'mode' can be supplied via command line arguments.
mode=0 means no chance constraint

mode=1 means gaussian

mode=2 means distributionally robust.

For example, to run the code with mode 1, type 'python MLFsim.py 1' in a terminal/command line (without quotes)

To run with mode 0, no argument needs to be supplied, so simply type 'python MLFsim.py', but 'python MLFsim.py 0' would also work.

I have hardcoded some ranges, and the output will be shown.
