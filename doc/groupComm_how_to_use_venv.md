# Intra Group Communication
## How to build and activate a Python venv


1. check if your current directory is `<path>/<to>/tempo-lite/`, if not, please make sure your current working directory is under `tempo-lite`. If you cannot see your current directory, simply try typing `pwd` in your terminal and read the output.

1. In your terminal, type \
`python3 -m venv venv` \
This command will automatically let Python build a folder containing the virtual environment under your current directory (which should be `<path>/<to>/tempo-lite/`)

1. check the `venv` folder has been constructed successfully. If you did not get any output after the previous command, it means it was successful, otherwise it would tell you it was not successful and give you some suggestions about how to fix the problem.

1. Try running the venv by typing \
`source venv\bin\activate` \
in the terminal. 

1. check if your virtual env was activated successfully. If there is any output, read the output carefully and see if it has offered a fix. If nothing was printed, and now in your terminal directory start with a `(venv)/<path>/<to>/tempolite$`, congratulations, you have sucessfully activated your virtual environment. 

1. now deactivate your venv by typing \
`deactivate` \
into your terminal
1. now run \
`venv/bin/pip install -r requirement.txt` \
**This could take a while**. PyTorch is a huge library.

1. if the previous step is successful, you have successfully build the environment we needed for this course project. Congratulations🎉, and thank you very much for your cooperations. 

1. *for next time and in the future*, just simply type `source venv/bin/activate` in under `tempo-lite/` and you will be good to go. You do not need to re-install your dependecies every time. 


