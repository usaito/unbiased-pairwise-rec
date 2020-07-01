import subprocess
from pathlib import Path

project_id = "neat-glazing-257206"
local_path = str(Path('.').resolve().parent)
gs_path = "trial:/home/elpistolero317/workspace/ubpr-real/logs"

if __name__ == "__main__":
    subprocess.run(['gcloud', 'compute', 'scp', '--project',
                    project_id, '--recurse', gs_path, local_path])
