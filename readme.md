Project developed under Ubuntu 20.04.5 LTS (GNU/Linux 5.15.90.1-microsoft-standard-WSL2 x86_64)
Should work on most recent standard linux distros, and windows using wsl2.
Obscure operating systems and arm-based processors will require manual setup, good luck !

After downloading, do the following:
* If you don't have it already, install conda virtual environment manager (https://docs.conda.io/projects/conda/en/stable/user-guide/index.html)
* Clone the conda virtual environment using **conda create --name saccade_env --file spec-file.txt**
* Activate this environment **conda activate saccade_env** (you will need to do it anytime you launch the code, it will change the command prompt)


Notes of to be done:
* rerun full "generalization_n_lines" train + test + aggregators with "oracle actions" for exploration for 500 steps, just to be sure it does not kill our conclusions ! 

It's a small hypothesis, mostly helps making sure network sees relevant states before it starts converging