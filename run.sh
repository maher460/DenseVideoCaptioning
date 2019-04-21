mkdir venv
virtualenv --system-site-packages -p python2.7 ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu
deactivate
