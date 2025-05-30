	sudo apt remove -y python3-torch-cuda python3-torchvision-cuda
	sudo apt purge -y python3-torch-cuda python3-torchvision-cuda
	sudo apt remove -y python3-tensorflow-cuda
	sudo apt purge -y python3-tensorflow-cuda
	sudo apt autoremove --purge -y
	pip3 uninstall -y torch torchvision torchaudio
	pip3 uninstall -y tensorflow tensorflow-gpu tf-keras
	sudo rm -rf /usr/lib/python3*/dist-packages/torch*
	sudo rm -rf /usr/local/lib/python3*/dist-packages/torch*
	sudo rm -rf ~/.local/lib/python3*/site-packages/torch*
	sudo rm -rf ~/.cache/torch
	sudo rm -rf ~/.torch
	sudo rm -rf /tmp/torch*
	sudo rm -rf /usr/lib/python3*/dist-packages/tensorflow*
	sudo rm -rf /usr/local/lib/python3*/dist-packages/tensorflow*
	sudo rm -rf ~/.local/lib/python3*/site-packages/tensorflow*
	sudo rm -rf ~/.local/lib/python3*/site-packages/tf_keras*
	sudo rm -rf ~/.cache/tensorflow
	sudo rm -rf /tmp/tensorflow*