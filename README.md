# CatConjunctivitisDetector
ENSE 489 - Social and Economic Impacts of Artificial Intelligence Project

If the environment isn't starting, do the following:
1. Open PowerShell as an administrator. To do this, right-click on the PowerShell icon and select "Run as administrator."

2. In the elevated PowerShell window, run the following command to check the current execution policy:
```
Get-ExecutionPolicy
```
3. If the output is "Restricted," it means that scripts are currently disabled. To enable script execution, run the following command:
```
Set-ExecutionPolicy RemoteSigned
```
This command allows the execution of local scripts but requires that all downloaded scripts be signed by a trusted publisher.

4. You will be prompted to confirm the change. Type "Y" and press Enter.

5. After setting the execution policy, try running the venv\Scripts\activate command again in your PowerShell window.

By adjusting the execution policy, you should be able to activate your virtual environment successfully.

### Step 1 
create the environment 
```
python3 -m venv venv
```

### Step 2
to activate the environment 
```
venv\Scripts\activate
```

### Step 3
install the libraries 
```
pip install opencv-python keras tensorflow numpy scikit-learn
```

### Step 4
The model is already trained so you can move onto Step 3, however if you want to train it yourself you can do so by:
```
python trainModel.py
```
### Step 5
Run the main.py file to detect if inputted image has conjunctivitis or not
```
python main.py
```
