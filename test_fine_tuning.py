import subprocess
import pytest
import glob
import os

def fine_tune(config_path):
	"""Run fine_tune.py with config_path

	Args:
		config_path: str
	"""
	process = subprocess.run(["python", "fine_tune.py","--config", config_path], text=True)

	assert process.returncode == 0


# config_paths  = glob.glob(os.path.join("test_configs", '*'))
config_paths = ['gpt2_lora_csv_noval.yaml']

@pytest.mark.parametrize("config_path", config_paths)
def test(config_path):
	"""Run test

	Args:
		config_path: str
	"""
	fine_tune(config_path)