# Set up your environment

Use **Python 3.11** and the pinned CPU environment below. Linux is the tested
platform; on Windows, use WSL2 with a Linux Python environment. No GPU, paid
cloud account or quantum computer is required.

Run these commands from your cloned group repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r kit/requirements-cpu.lock
python -m pip install -r requirements-notebook.txt -c kit/requirements-cpu.lock
python -m pip check
python -c "from kit.sdk import verify_assets; print(verify_assets()['status'])"
python -m unittest discover -s tests -v
python -m jupyterlab QAML_CHALLENGE.ipynb
```

The integrity check prints `released`. Select the Python kernel from this
environment. If you use an existing Jupyter installation, register the kernel:

```bash
python -m ipykernel install --user --name qaml-2026 --display-name 'QAML 2026 (Python 3.11)'
```

Select **QAML 2026 (Python 3.11)** in the notebook. Run cells in order. Local
examples use the included files; the black-box cell makes a request only after
you explicitly enable it. Keep the notebook private and clear outputs before
sharing it. Never type a token directly into a code cell.

If an integrity check fails, restore the released kit from your group's starter
commit; do not change hashes to silence the error. Put your code and generated
artifacts in your own directories rather than editing `kit/`.

See [Jupyter installation](https://jupyter.org/install) and [black-box access](black-box-access.md).
