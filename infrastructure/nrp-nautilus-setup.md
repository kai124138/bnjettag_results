# NRP / Nautilus — Setup & Workflow Notes

My working reference for connecting to and using the NRP Nautilus cluster. This uses the current **kubelogin** login flow and replaces the older "download a config with a token baked in" method (the one that kept throwing `401 Unauthorized` and needing a re-download).

**Namespace:** `cms-ml` — the Duarte group's *shared* namespace (Russell added me).

---

## One-time setup

**1. Install kubectl**
```
brew install kubectl
```

**2. Install the kubelogin plugin** — the binary must end up named `kubectl-oidc_login` on your PATH. Pick one:
- krew (recommended): `kubectl krew install oidc-login`
- Homebrew tap: `brew install int128/kubelogin/kubelogin`
  - ⚠️ NOT plain `brew install kubelogin` — that's a *different* (Azure) tool with the same name.
- Manual: download the release from int128/kubelogin, rename to `kubectl-oidc_login`, move into your PATH.

Verify: `kubectl oidc-login --version`

**3. Download the Nautilus config** to `~/.kube/config`:
```
mkdir -p ~/.kube
curl -o ~/.kube/config -fSL "https://nrp.ai/config"
```
This config already wires up oidc-login, so there's nothing else to configure.

**4. Set the context and log in** — the first kubectl command opens a browser for CILogon:
```
kubectl config use-context nautilus
kubectl get nodes
```
Pick **UC San Diego**, log in with UCSD creds. Your account is created on first login.

**5. Namespace access** — students can't self-add. Russell (or Javier) adds you to `cms-ml` at https://nrp.ai/namespaces. Verify:
```
kubectl get pods -n cms-ml
```
`No resources found` = you're in (just no pods of your own yet).

**6. (Optional) Make cms-ml the default** so you stop typing `-n`:
```
kubectl config set-context --current --namespace=cms-ml
```

**Why kubelogin is better:** the access token auto-refreshes (expires ~30 min, renews automatically), so no more manually re-downloading the config when it dies.

---

## Everyday commands

```
kubectl get pods -n cms-ml                       list pods
kubectl create -f mypod.yaml -n cms-ml           create a pod from yaml
kubectl describe pod <pod> -n cms-ml             details / events (debugging)
kubectl logs <pod> -n cms-ml                     logs  (add -f to follow live)
kubectl exec -it <pod> -n cms-ml -- /bin/bash    shell into a running pod
kubectl delete pod <pod> -n cms-ml               delete a pod
kubectl oidc-login clean                         force token refresh (e.g. after joining a new namespace)
```
Replace `<pod>` with the real pod name — no angle brackets.

---

## Minimal test pod

`kai-test-pod.yaml`:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kai-test-pod
spec:
  restartPolicy: Never
  containers:
  - name: kai-test
    image: ubuntu:22.04
    resources:
      limits:
        memory: 200Mi
        cpu: 200m
      requests:
        memory: 200Mi
        cpu: 200m
    command: ["sh", "-c", "echo connected && sleep 3600"]
```

Run **one command at a time**:
```
kubectl create -f kai-test-pod.yaml -n cms-ml
kubectl get pods -n cms-ml
```
Wait until `kai-test-pod` shows `Running`, then shell in:
```
kubectl exec -it kai-test-pod -n cms-ml -- /bin/bash
```
You land at `root@kai-test-pod:/#`. Type `exit`, then clean up:
```
kubectl delete pod kai-test-pod -n cms-ml
```

---

## Gotchas I actually hit (and fixes)

- **`zsh: parse error near '\n'`** — left the literal placeholder `<namespace>` in the command. `<` is a redirect operator in zsh. Fix: replace `<namespace>` with the real name (`cms-ml`), no angle brackets.

- **`Error from server (NotFound): pods "wait" not found`** (also `"for"`, `"to"`, etc.) — zsh doesn't treat `#` as a comment by default, so pasted inline `# comments` got passed to kubectl as pod names. Fixes: don't paste comments, **or** add `setopt interactive_comments` to `~/.zshrc` so `#` is ignored even in pasted blocks.

- **`unable to upgrade connection: container not found`** on exec — ran `exec` before the pod finished starting (`ContainerCreating` = still pulling the image). Fix: wait for `Running` before exec.

- **Multi-line pastes** — run kubectl commands one at a time (or use the `setopt` fix). Pasting a whole block can chain failures.

- **`Forbidden`** — authenticated but not a member of that namespace. Get added at nrp.ai/namespaces.

- **`401 Unauthorized` / token errors** — `kubectl oidc-login clean`, then re-run any kubectl command.

---

## Shared namespace etiquette

`cms-ml` is the **whole Duarte group's** namespace, so most pods you see belong to other people (training sweeps, `*serverdep` volume web-servers, the `mpt-*` monitoring stack).

- **Prefix your resources** so they're obviously yours (others use `zh-`, `tn-`, `rino-`, `ajd-`… → I use `kai-`).
- **Only touch your own pods.** Never delete — and never *force*-delete — someone else's.
- **Containers are stateless.** Anything not on a persistent volume is gone when the pod restarts. Don't run never-ending `sleep` jobs (it can get you banned).

---

## Remote / headless machines (lxplus, Mulder)

No local browser → the auto-login won't open a window. Use the device-code flow:
```
--grant-type=device-code --skip-open-browser
```
It prints a URL to paste into your laptop's browser. Easiest is to do interactive work from your **laptop**, where the browser just opens.

---

## Next steps

1. **Persistent Volume (PVC)** — claim storage so code, data, and training outputs survive pod restarts.
2. **Get code + data onto the volume** — clone BNJetTagKai and stage the training data.
3. **GPU pod** mounted to the volume — request a GPU, pull the repo, run training.
4. **(Optional) Docker image** with my deps (qkeras / tensorflow / hls4ml) so I don't reinstall every time.
5. **Pods vs Jobs** — interactive pods for dev/testing; batch jobs for long training runs.
