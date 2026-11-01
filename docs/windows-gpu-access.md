# Windows GPU access for a research preflight

This note covers safe, practical SSH access and the current repository layout on
the reported Windows RTX 3050 laptop. It is not a training guide and does not
establish project model fit.

## Reported hardware and unverified points

- Reported target: NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB, driver
  581.83, on a Windows laptop.
- Verified: user-authorized SSH access, Windows 11 AMD64, Python 3.13.6, and
  CUDA availability through the installed PyTorch environment.
- Unverified: available GPU time and whether the project pilot or any batch size
  fits. The earlier generic CUDA smoke does not establish project pilot fit.
- 6 GB of VRAM is not a fit guarantee. Capacity must be measured.
- No GPU-time budget, project training, or validation is assumed here.

## Prerequisites on the Windows laptop

- **OpenSSH Server:** install and start the Windows OpenSSH Server, and confirm
  the SSH service is running. Keep the firewall and other protections enabled;
  use a scoped allow rule for SSH if one is needed.
- **Key-based authentication:** use an existing key pair or create one, place
  the public key in the account's `authorized_keys`, and keep the private key on the client.
  Confirm that login works without sending a password.
- **Repository:** the public repository is cloned into the `source`
  subdirectory of the project folder. The existing isolated `.venv` remains in
  its parent.
- **Network reachability:** identify the laptop's hostname or IP address and SSH
  port, and confirm the route from the client, for example over the same LAN or
  an approved VPN.
- **Power and availability:** keep the laptop plugged in and prevent it from
  sleeping during the agreed window. Confirm with the owner that the GPU is free
  for that window and how much time is actually available.

## What to share, and what not to share

Share only:

- hostname or IP address;
- SSH port, if it is not the default;
- username;
- confirmation that key-based login works.

Do **not** send a private key, password, recovery code, or other secret in chat.
Do not put credentials in the repository. If a password prompt appears, key-based
authentication is not configured yet; fix that on the laptop rather than sending
the password.

The Codex environment may need network permission before it can reach the laptop.

## Repository and environment layout

At the user's request, the GitHub repository was made public and cloned into
the `source` subdirectory of the project folder on the Windows laptop. The
existing isolated `.venv` remains in the parent project folder. From PowerShell
in the clone, set the project environment and synchronize with:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "..\.venv"
..\.venv\Scripts\python.exe -m uv sync --locked --inexact --extra train
```

Invoke project modules with `..\.venv\Scripts\python.exe -m ...`, for example
`..\.venv\Scripts\python.exe -m exposure.pilot_train`. Locked `uv sync`, 123
Windows tests, and a short project CUDA run completed.

## Run the preflight on the Windows machine

From the cloned `source` directory on the Windows laptop, in PowerShell:

```powershell
..\.venv\Scripts\python.exe -m exposure.gpu_preflight
```

The command reports:

- the operating system and Python version;
- the NVIDIA driver and GPU information;
- whether PyTorch with CUDA is available.

It does not install packages, change drivers, download data, or train a model.
Share the command output, not credentials.

## Safety and interpretation

- SSH access and CUDA availability are established, but project model fit,
  available GPU time, and validation remain unverified.
- A preflight pass is not a project-fit result, a Stage 0 gate, or a budget for
  an unbounded training run.
- For the broader plan, see
  [research and hardware plan](research-and-hardware-plan.md) and the
  [human checklist](human-checklist.md).
