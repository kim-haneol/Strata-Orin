# Gyro-M 262K validation and persistent research Hermes deployment

Status: completed. Context/API validation, persistent services, recovery and research integration passed. User authorized validation, persistent service
deployment and research Hermes migration on port 22868. The later correction
explicitly preserves strataorin; no profile removal is authorized.

## Frozen protocol

- Existing f16cache binary `996f228f4e3b16c61ba804589f3053b678a9401c5837159a2f2e44fefd4b3256`,
  `mtp/rt-enko`, INT8 KV, 24576 expert slots, owned 8192-token prefill chunks.
- Change only max-context to 262144 for initial validation; same capacity for every request.
- Short Korean arithmetic smoke, then exact rendered input lengths 64000, 128000,
  253952; at most 512 output tokens, leaving 8192 tokens at the largest input.
- Independent deterministic synthetic filler for each length. Recover three exact
  identifiers near beginning/middle/end; validate server-reported prompt count.
  This is a bounded recall check, not a broad long-context quality evaluation.
- API tool call + result continuation; A/B/A/B conversation switching with cache
  reuse; English SSE stream including usage and terminal marker.
- Actual research Hermes reads/edits a disposable absolute-path fixture after
  model-server validation. No messaging tools are invoked.
- ASTOP named watches and native raw recording; preserve before/after state,
  minimum available RAM, swap, GPU activity/clocks/temperature. Unavailable stays null.
- Explicit task exit and result artifacts determine success, not unknown ASTOP exits.
- Permanent services only after capacity/recall/API checks pass. Host loopback
  22868 forwards over USB NCM to Jetson loopback 18092. Verify restart/reconnect.
  Also preserve host loopback 18092 in the same SSH connection for strataorin.

Raw requests, engine logs and telemetry remain in ignored `work/` and on Jetson;
manifest, result summary, deployment files and rollback instructions are committed.

## Deployment and rollback contract

- Jetson user `jetson`, persistent `jetson-gyro-api.service`, enabled under
  `default.target`; user lingering is already enabled. It runs the existing
  `serve-strata-gyro.sh` with the runtime `server.json`.
- Host user `haneol`, persistent `jetson-gyro-tunnel-22868.service`, also enabled
  under `default.target`; lingering is already enabled. A single USB NCM SSH
  connection forwards both loopback ports 22868 and 18092 to Jetson 18092.
- Research `default` keeps provider `strata-local`, endpoint
  `http://127.0.0.1:22868/v1`, context 262144 and output cap 8192. Only three model
  identifier occurrences are changed to `qwen38-flash-next-gyro-m`; comments,
  unrelated settings, credentials and gateway integration settings are preserved.
- Preserve `${HERMES_PROFILE}` and its original config
  SHA256 `43ae62ff88b88e9a9e874fd56223b45a2935528522d692b7c406f069c369b3eb`.

Rollback, if needed:

1. As `research`, restore `.hermes/config.yaml.before-gyro-20261009` over
   `.hermes/config.yaml`, then restart `hermes-gateway.service`. Its previous
   endpoint 22868 was unavailable; restoring the config does not restore the old model server.
2. On Jetson, stop `jetson-gyro-api.service`, restore runtime
   `server.json.before-262k-20261009` over `server.json`, then start the API service
   to serve the previously deployed 64K f16cache configuration.
3. To remove this task's persistent forwarding, disable/stop the host
   `jetson-gyro-tunnel-22868.service`. The original transient 18092 forwarding can
   be recreated with `systemd-run --user --unit=jetson-strata-tunnel-18092` and the
   same SSH command with only `-L 127.0.0.1:18092:127.0.0.1:18092`.

## Preparation incident

The first validation client started while the engine was still loading and exited
1 with connection refused, before any generation. It is retained as
`result-startup-race.json`; bounded API readiness was added before the actual run.
All three frozen exact-token requests and their hashes were recorded.

## Context/API results

All 11 checks passed (`summary.json`). All three long requests were cold
(`cache_n=0`) and the server reported exactly the frozen rendered input counts.

| Actual input tokens | HTTP wall seconds | Internal prefill seconds | Internal decode tok/s | Dispersed identifiers |
|---:|---:|---:|---:|---|
| 64,000 | 114.3 | 112.6 | 32.4 | 3/3 |
| 128,000 | 235.0 | 233.0 | 31.9 | 3/3 |
| 253,952 | 496.7 | 494.2 | 30.8 | 3/3 |

The maximum-length request reserves 8192 output tokens in the configured capacity;
this test generated a short recall answer, not 8192 output tokens. Rates are for
synthetic structured filler and recall, not representative agent decode throughput.
The API tests include one tool call/result cycle, two conversation switches with
positive cached-token counts, and a completed English SSE stream with usage.

ASTOP recording (`telemetry-summary.json`): 454 snapshots, maximum used memory
61,460,467,712 bytes (57.2 GiB) out of 61.4 GiB; about 4.2 GiB minimum headroom
by total minus used. Pressure states included normal and warning. Swap remained
451,469,312 bytes throughout. Maximum GPU temperature 63.687 C, observed frequency
up to 1300.5 MHz. GPU power and process GPU memory were unavailable, not zero.

Completion authority: every HTTP response and expected prompt count/identifier was
validated again by a separate artifact checker, whose direct exit was 0
(`artifact-validator-exit-code.txt`). The original ASTOP terminal event is
`disappeared/unknown`; it is retained and is not treated as success. The successful
transient validation unit had already been garbage-collected before inspection,
so the default zero fields in `unit-exit.txt` are not exit-code authority.

Host service control uses `systemctl --user --machine=haneol@.host`: the default
local private manager socket refused connections, while the host manager route
worked. No user manager or unrelated service was restarted.

## Research integration

The actual research default profile completed the absolute-path file read/write
fixture, with sum 42, exact marker `ORIN_HERMES_6149`, and `status=verified`.
`hermes-usage.json` records model `qwen38-flash-next-gyro-m`, 4 API calls, 494
output tokens and 15,719 cache-read tokens. The retained `RemainAfterExit=yes`
integration unit proves `ExecMainCode=1` (CLD_EXITED), `ExecMainStatus=0`,
`LoadState=loaded`, `SubState=exited` (`hermes-unit-exit.txt`). The ASTOP exit
status is independently unknown and is not substituted for that direct receipt.

The tunnel was externally terminated with SIGTERM and automatically restarted:
PID changed 1304170 -> 1307733, `NRestarts=1`, enabled/active, and a real Korean
generation probe passed (`tunnel-recovery.json`). The profile config SHA256 for
strataorin is unchanged. No profile, replay, model or request data was deleted.

## Final operating state

- Jetson `jetson-gyro-api.service`: enabled/active, f16cache + enko, INT8 KV,
  maximum context 262144, local listener 18092; config hash `19c37385...`.
- Host `jetson-gyro-tunnel-22868.service`: enabled/active, USB NCM SSH,
  both localhost 22868 and 18092 forward to Jetson localhost 18092.
- Both user accounts already have lingering enabled for boot/logout operation.
- Server termination/recovery passed: PID 374481 -> 374883, `NRestarts=1`,
  loaded API and real generation returned after 76.4 seconds
  (`server-recovery.json`). Tunnel recovery also passed in 7.6 seconds.
- Research default: `qwen38-flash-next-gyro-m`, `strata-local`, endpoint 22868,
  context 262144 and output cap 8192. `hermes-gateway.service` restarted with
  PID 1310466, enabled/active; its own state records Telegram connected,
  no error and session store OK (`gateway-state.json`). No test message was sent.
- Both API ports report loaded Gyro-M/262144 (`final-health-*.json`).
- Strataorin directory/config is preserved byte for byte; its 18092 connection
  remains available. No deletion was performed.
- Final artifact/config/health/recovery/profile-hash checks passed. Raw ASTOP
  terminal events were recorded before acknowledgment, and only this task's
  watches were removed. The production model server and tunnel remain running.

Boot enablement/lingering and process recovery were verified. Neither computer
was rebooted, and the USB cable was not physically disconnected. Long-context
quality beyond these three synthetic identifiers, and multiple simultaneous
maximum-length conversations, are outside the validation scope.
