This code is based on this repository: https://github.com/kvablack/LLaVA-server

### 1. Create a new environment and install the dependencies listed in requirements.txt
### 2. Run a llava server:
```
gunicorn "app:create_app()" &
```
Before running the above command, update the ckpt path at line 63 in:
```
Curriculum-DPO/reward_models/llava-server/llava_server/llava.py
```