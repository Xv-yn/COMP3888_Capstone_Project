# Title

TODO: Project Summary Goes Here

## Deployment

##### Hardware Requirements

##### Software Requirements

- Python3.10

##### Setting up a Virual Environment

###### Python venv

1. Clone and move to the repository directory

```bash
git clone https://github.com/Xv-yn/COMP3888_Capstone_Project
cd COMP3888_Capstone_Project
```

2. Create the virtual environment

```bash
python -m venv venv
```

3. Activate the virtual environment
   - Linux

   ```bash
   source venv/bin/activate
   ```

4. Install required packages

```bash
pip install requirements.txt
```

5. Finished setting up virtual environment

##### Running the Application

TODO: Make easier accessible to user

As of now:

```bash
cd animal_action
python tools/top_down_video_det.py ./path/to/img [--no-skeleton]
```

Sample usage:

- Generate full image (with skeleton)

  ```bash
  python tools/top_down_video_det.py examples/img3.png
  ```

- Generate image without skeleton
  ```bash
  python tools/top_down_video_det.py examples/img3.png --no-skeleton
  ```
