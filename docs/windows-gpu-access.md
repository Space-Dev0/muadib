# Windows GPU access for a research preflight

This note covers safe, practical SSH access to the reported Windows RTX 3050
laptop for a **GPU preflight only**. It is not a training setup guide and does
not establish that the machine is reachable or suitable.

## Reported hardware and unverified points

- Reported target: NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB, driver
  581.83, on a Windows laptop.
- Unverified: network reachability, SSH access, Windows edition, CUDA/PyTorch
  compatibility, available GPU time, and whether any model or batch size fits.
- 6 GB of VRAM is not a fit guarantee. Capacity must be measured; a successful
  login does not prove CUDA works.
- No access, budget, installation, training, or validation is assumed here.

## Prerequisites on the Windows laptop

- **OpenSSH Server:** install and start the Windows OpenSSH Server, and confirm
  the SSH service is running. Keep the firewall and other protections enabled;
  use a scoped allow rule for SSH if one is needed.
- **Key-based authentication:** use an existing key pair or create one, place
  the public key in the account's `authorized_keys`, and keep the private key on the client.
  Confirm that login works without sending a password.
- **Account and repository:** use an account that can reach the repository and
  run Python from it.
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

## Run the preflight on the Windows machine

From the repository on the Windows laptop, in PowerShell or a terminal:

```powershell
python -m exposure.gpu_preflight
```

The command reports:

- the operating system and Python version;
- the NVIDIA driver and GPU information;
- whether PyTorch with CUDA is available, if PyTorch is already installed.

It does not install packages, change drivers, download data, or train a model.
Share the command output, not credentials.

## Safety and interpretation

- Actual reachability and GPU/PyTorch compatibility remain unverified until the
  preflight is run on the laptop.
- A preflight pass is not a hardware-fit result, a Stage 0 gate, or permission
  for an unbounded training run.
- For the broader plan, see
  [research and hardware plan](research-and-hardware-plan.md) and the
  [human checklist](human-checklist.md).
